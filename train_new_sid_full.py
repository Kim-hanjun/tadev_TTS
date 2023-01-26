import logging
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler

from network import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)

import utils
from utils.data_utils import TextAudioSpeakerDataset, TextAudioSpeakerCollate, DistributedBucketSampler
from utils.text_utils import SYMBOLS

from train import train


numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


torch.backends.cudnn.benchmark = True
global_step = 0

GENERATOR_CHECKPOINT_FILENAME = "/home/hun/workspace/tadev-TTS/logs/ver1_56/G_853000.pth"
DISCRIMINATOR_CHECKPOINT_FILENAME = "/home/hun/workspace/tadev-TTS/logs/ver1_56/D_853000.pth"


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "80000"

    args = utils.get_args()
    log_dir_path = args.log_dir_path
    config_filename = args.config_filename
    model_name = args.model_name

    model_dir_path = os.path.join(log_dir_path, model_name)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    hps = utils.get_hparams(config_filename)
    hps.model_dir_path = model_dir_path

    mp.spawn(
        main_worker,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def main_worker(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir_path)
        logger.info(hps)
        writer_train = SummaryWriter(log_dir=os.path.join(hps.model_dir_path, "train"))
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir_path, "eval"))
    else:
        logger = None
        writer_train = None
        writer_eval = None

    dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerDataset(
        filepath_sid_text_filename=hps.data.training_files,
        sr=hps.data.sampling_rate,
        max_wav_value=hps.data.max_wav_value,
        n_fft=hps.data.filter_length,
        hop_length=hps.data.hop_length,
        win_length=hps.data.win_length,
        n_mels=hps.data.n_mel_channels,
        fmin=hps.data.mel_fmin,
        fmax=hps.data.mel_fmax,
        min_text_len=hps.data.min_text_len,
        max_text_len=hps.data.max_text_len,
        add_blank=hps.data.add_blank,
    )

    train_sampler = DistributedBucketSampler(
        train_dataset, hps.train.batch_size, hps.data.boundaries, num_replicas=n_gpus, rank=rank, shuffle=True
    )
    collate_fn = TextAudioSpeakerCollate()
    train_dl = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerDataset(
            filepath_sid_text_filename=hps.data.validation_files,
            sr=hps.data.sampling_rate,
            max_wav_value=hps.data.max_wav_value,
            n_fft=hps.data.filter_length,
            hop_length=hps.data.hop_length,
            win_length=hps.data.win_length,
            n_mels=hps.data.n_mel_channels,
            fmin=hps.data.mel_fmin,
            fmax=hps.data.mel_fmax,
            min_text_len=hps.data.min_text_len,
            max_text_len=hps.data.max_text_len,
            add_blank=hps.data.add_blank,
        )
        eval_dl = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    else:
        eval_dl = None

    net_g = SynthesizerTrn(
        n_vocab=len(SYMBOLS),
        spec_channels=hps.data.filter_length // 2 + 1,
        segment_size=hps.train.segment_size // hps.data.hop_length,
        inter_channels=hps.model.inter_channels,
        hidden_channels=hps.model.hidden_channels,
        filter_channels=hps.model.filter_channels,
        n_heads=hps.model.n_heads,
        n_layers=hps.model.n_layers,
        kernel_size=hps.model.kernel_size,
        dropout_ratio=hps.model.dropout_ratio,
        resblock=hps.model.resblock,
        resblock_kernel_sizes=hps.model.resblock_kernel_sizes,
        resblock_dilation_sizes=hps.model.resblock_dilation_sizes,
        upsample_rates=hps.model.upsample_rates,
        upsample_initial_channel=hps.model.upsample_initial_channel,
        upsample_kernel_sizes=hps.model.upsample_kernel_sizes,
        n_speakers=hps.new_sid.pretrained_n_speakers,
        gin_channels=hps.model.gin_channels,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

    net_g, _, _ = utils.load_checkpoint(GENERATOR_CHECKPOINT_FILENAME, net_g, logger=logger)
    net_g = utils.expand_emb_g(net_g, hps.new_sid.new_n_speakers)
    net_d, _, _ = utils.load_checkpoint(DISCRIMINATOR_CHECKPOINT_FILENAME, net_d, logger=logger)
    # utils.freeze_layers(net_g, ["emb_g.weight"])

    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    epoch_start = 1
    global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_start - 2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_start - 2)

    grad_scaler = GradScaler(enabled=hps.train.fp16_run)

    train(
        rank=rank,
        model_dir_path=hps.model_dir_path,
        epoch_start=epoch_start,
        n_epochs=hps.train.n_epochs,
        logger=logger,
        writer_train=writer_train,
        writer_eval=writer_eval,
        net_g=net_g,
        net_d=net_d,
        optim_g=optim_g,
        optim_d=optim_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        grad_scaler=grad_scaler,
        train_dl=train_dl,
        eval_dl=eval_dl,
        segment_size=hps.train.segment_size,
        n_fft=hps.data.filter_length,
        hop_length=hps.data.hop_length,
        win_length=hps.data.win_length,
        n_mels=hps.data.n_mel_channels,
        sampling_rate=hps.data.sampling_rate,
        fmin=hps.data.mel_fmin,
        fmax=hps.data.mel_fmax,
        center=False,
        mel_loss_weight=hps.train.c_mel,
        kl_loss_weight=hps.train.c_kl,
        log_interval=hps.train.log_interval,
        eval_interval=hps.train.eval_interval,
        fp16_run=hps.train.fp16_run,
    )


if __name__ == "__main__":
    main()
