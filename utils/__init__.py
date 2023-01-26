import os
import sys
import argparse
import glob
import torch
import torch.nn.functional as F
import numpy as np
import logging
import json
from typing import Union, List


MATPLOTLIB_FLAG = False


def to_float_tensor(data: Union[list, np.array]):
    return torch.FloatTensor(data.astype(np.float32))


def get_logger(log_dir_path: str, log_filename: str = "training_log.log"):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging
    logger = logging.getLogger(os.path.basename(log_dir_path))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    file_handler = logging.FileHandler(os.path.join(log_dir_path, log_filename))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir_path", type=str, default="./logs/")
    parser.add_argument(
        "-c",
        "--config_filename",
        type=str,
        required=True,
        help="File name including path of JSON file for configuration",
    )
    parser.add_argument("-m", "--model_name", type=str, required=True, help="Model name")
    return parser.parse_args()


# For Hyper Parameters


class HyperParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HyperParams(**v)  # use recursively
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_hparams(config_filename: str):
    with open(config_filename, "r") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HyperParams(**config)

    return hparams


# For Network


def slice_3d_sequence(x: torch.Tensor, indices_start: torch.Tensor, segment_size: int = 4):
    """torch.Tensor를 받아 일부를 자름.

    Args:
        x(torch.Tensor): torch.Tensor for slicing. [batch_size, hidden_ch, sequence_length]
        indices_start(torch.Tensor): first indices of segments. [batch_size]
        segement_size(int): length of segment(ret)

    Returns:
        ret: segment result. [batch_size, hidden_ch, segment_size]
    """
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        index_start = indices_start[i]
        index_end = index_start + segment_size
        ret[i] = x[i, :, index_start:index_end]
    return ret


def rand_slice_3d_sequence(x: torch.Tensor, x_lengths: torch.Tensor = None, segment_size: int = 4):
    """torch.Tensor를 받아 일부를 random하게 자름. VAE의 latent variable `z`를 자르는 용도.

    Args:
        x(torch.Tensor): torch.Tensor for slicing. [batch_size, hidden_ch, sequence_length]
        x_length(torch.Tensor): length of x. [batch_size]
        segement_size(int): length of segment(ret)

    Returns:
        ret: segment result. [batch_size, hidden_ch, segment_size]
        indices_start: first indices of segments. [batch_size]
    """
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    indices_start = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
    ret = slice_3d_sequence(x, indices_start, segment_size)
    return ret, indices_start


def make_mask(length: torch.Tensor, max_length: int = None):
    """batch의 element들의 sequence_length로부터 mask를 만든다.

    args:
        length(torch.Tensor): [batch_size]
        max_length: 해당 batch안의 element들 중 max_length

    returns:
        mask: True, False로 이루어진 mask. [batch_size, max_length]
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    mask = x.unsqueeze(0) < length.unsqueeze(1)
    return mask


def convert_pad_shape(pad_shape):
    inverse = pad_shape[::-1]
    pad_shape = [item for sublist in inverse for item in sublist]  # flatten list
    return pad_shape


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """

    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = make_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


# For Tensorboard


def get_grad_values(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    norm_type = float(norm_type)

    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)  # sum(grad ** 2) ** 0.5
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def add_tensorboard_items(
    writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


# For checkpoint


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim, iteration: int, checkpoint_filename: str, logger=None
):
    """Save model state dict, epoch, optimizer and learning rate.

    Args:
        model: torch model.
        optimizer: torch optimizer.
        iteration: epoch.
        checkpoint_filename: checkpoint filename including path.
    """

    if logger is not None:
        logger.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_filename))
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({"model": state_dict, "iteration": iteration, "optimizer": optimizer.state_dict()}, checkpoint_filename)


def load_checkpoint(checkpoint_filename: str, model: torch.nn.Module, optimizer: torch.optim = None, logger=None):
    """Load model checkpoint.

    Args:
        checkpoint_filename: checkpoint filename including path.
        model: torch model.
        optimizer: torch optimizer.

    Returns:
        model: model loaded.
        optimizer: optimizer loaded.
        iteration: epoch.
    """

    assert os.path.isfile(checkpoint_filename)
    checkpoint_dict = torch.load(checkpoint_filename, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}  # # 굳이 new로 새로 만들어야하나. 그냥 saved_state_dict로 load해도 될 것 같은데.
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except Exception:
            # logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    if logger is not None:
        logger.info("Loaded checkpoint '{}' (iteration {})".format(checkpoint_filename, iteration))
    return model, optimizer, iteration


def latest_checkpoint_path(checkpoint_dir_path: str, regex: str = "G_*.pth"):
    """Return latest checkpoint filename.

    Args:
        checkpoint_dir_path: dir path where checkpoints were saved.
        regex: regular expression of checkpoint file.
    """

    f_list = glob.glob(os.path.join(checkpoint_dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    latest_checkpoint_filename = f_list[-1]
    return latest_checkpoint_filename


# For plotting


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt  # # pyplot으로 바꾸는게 좋을 듯?
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


# For embedding


def expand_emb_g(model: torch.nn.Module, n_new_sids: int = 1):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    emb_weight = state_dict["emb_g.weight"]
    new_device = emb_weight.device
    n_sids, emb_dim = emb_weight.size()
    emb_weight = torch.cat([emb_weight, torch.randn(n_new_sids, emb_dim).to(new_device)], axis=0)
    # new_emb_weight = emb_weight.mean(axis=0).repeat(n_new_sids, 1)
    # emb_weight = torch.cat([emb_weight, new_emb_weight], axis=0)
    state_dict["emb_g.weight"] = emb_weight

    if hasattr(model, "module"):
        model.module.emb_g = torch.nn.Embedding(n_sids + n_new_sids, emb_dim).to(new_device)
        model.module.load_state_dict(state_dict)
    else:
        model.emb_g = torch.nn.Embedding(n_sids + n_new_sids, emb_dim).to(new_device)
        model.load_state_dict(state_dict)
    return model


def init_new_sid_emb_g(model: torch.nn.Module, n_new_sids: int = 1):
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    emb_weight = state_dict["emb_g.weight"]
    emb_dim = emb_weight.size(1)
    new_device = emb_weight.device
    emb_weight = emb_weight.mean(axis=0).repeat(n_new_sids, 1)
    state_dict["emb_g.weight"] = emb_weight

    if hasattr(model, "module"):
        model.module.emb_g = torch.nn.Embedding(n_new_sids, emb_dim).to(new_device)
        model.module.load_state_dict(state_dict)
    else:
        model.emb_g = torch.nn.Embedding(n_new_sids, emb_dim).to(new_device)
        model.load_state_dict(state_dict)
    return model


def expand_emb_g_with_new_emb_weight(model: torch.nn.Module, new_emb_checkpoint_filename: str):
    new_emb_g_weight = torch.load(new_emb_checkpoint_filename)["model"]["emb_g.weight"]
    n_new_sids = new_emb_g_weight.size(0)

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    emb_weight = state_dict["emb_g.weight"]
    new_device = emb_weight.device
    n_sids, emb_dim = emb_weight.size()
    emb_weight = torch.cat([emb_weight, new_emb_g_weight.to(new_device)], axis=0)
    state_dict["emb_g.weight"] = emb_weight

    if hasattr(model, "module"):
        model.module.emb_g = torch.nn.Embedding(n_sids + n_new_sids, emb_dim).to(new_device)
        model.module.load_state_dict(state_dict)
    else:
        model.emb_g = torch.nn.Embedding(n_sids + n_new_sids, emb_dim).to(new_device)
        model.load_state_dict(state_dict)
    return model


def freeze_layers(model, train_param_list):
    if hasattr(model, "module"):
        train_param_list = ["module." + name for name in train_param_list]
    for name, param in model.named_parameters():
        if name in train_param_list:
            param.requires_grad = True
        else:
            param.requires_grad = False
