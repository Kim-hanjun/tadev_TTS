import logging
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from network import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from losses import generator_loss, discriminator_loss, feature_loss, kl_loss

from typing import Union

import utils
from utils.data_utils import TextAudioSpeakerDataset, TextAudioSpeakerCollate, DistributedBucketSampler
from utils.text_utils import SYMBOLS


numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


torch.backends.cudnn.benchmark = True
global_step = 0


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
        n_speakers=len(train_dataset.sid_dict),
        gin_channels=hps.model.gin_channels,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

    try:
        net_g, optim_g, epoch_start = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir_path, "G_*.pth"), net_g, optim_g, logger
        )
        net_d, optim_d, epoch_start = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir_path, "D_*.pth"), net_d, optim_d, logger
        )
        global_step = (epoch_start - 1) * len(train_dl)
    except Exception:
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


def train(
    rank,
    model_dir_path,
    epoch_start,
    n_epochs,
    logger,
    writer_train,
    writer_eval,
    net_g,
    net_d,
    optim_g,
    optim_d,
    scheduler_g,
    scheduler_d,
    grad_scaler,
    train_dl,
    eval_dl,
    segment_size,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    sampling_rate: int,
    fmin: int,
    fmax: int,
    center: bool = False,
    mel_loss_weight: Union[int, float] = 45,
    kl_loss_weight: Union[int, float] = 1.0,
    log_interval: int = 200,
    eval_interval: int = 1000,
    fp16_run: bool = True,
):
    def train_one_epoch(epoch):

        global global_step
        train_dl.batch_sampler.set_epoch(epoch)  # needed for shuffling

        net_g.train()
        net_d.train()
        for batch_idx, batch in enumerate(train_dl):
            text, text_lengths, wav, wav_lengths, linspec, linspec_lengths, melspec, melspec_lengths, sid = batch
            text, text_lengths = text.cuda(rank, non_blocking=True), text_lengths.cuda(rank, non_blocking=True)
            wav, wav_lengths = wav.cuda(rank, non_blocking=True), wav_lengths.cuda(rank, non_blocking=True)
            linspec, linspec_lengths = linspec.cuda(rank, non_blocking=True), linspec_lengths.cuda(
                rank, non_blocking=True
            )
            melspec, melspec_lengths = melspec.cuda(rank, non_blocking=True), melspec_lengths.cuda(
                rank, non_blocking=True
            )
            sid = sid.cuda(rank, non_blocking=True)

            with autocast(fp16_run):
                # generator
                (
                    wav_sliced_hat,
                    l_length,
                    attn,
                    ids_slice,
                    text_mask,
                    linspec_mask,
                    (z, z_theta, mean_text, logs_text, mean_linspec, logs_linspec),
                ) = net_g(text, text_lengths, linspec, linspec_lengths, sid)
                wav_sliced = utils.slice_3d_sequence(wav, ids_slice * hop_length, segment_size)
                melspec_sliced_hat = utils.audio_utils.wav_to_melspec(
                    wav_sliced_hat, n_fft, hop_length, win_length, n_mels, sampling_rate, fmin, fmax, center
                )
                melspec_sliced = utils.slice_3d_sequence(melspec, ids_slice, segment_size // hop_length)
                # discriminator
                disc_real_hat, disc_gen_hat, _, _ = net_d(wav_sliced, wav_sliced_hat.detach())

                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(disc_real_hat, disc_gen_hat)

            optim_d.zero_grad()
            grad_scaler.scale(loss_disc).backward()
            grad_scaler.unscale_(optim_d)
            grad_norm_d = utils.get_grad_values(net_d.parameters(), None)
            grad_scaler.step(optim_d)

            with autocast(fp16_run):
                disc_real_hat, disc_gen_hat, fmap_real, fmap_gen = net_d(wav_sliced, wav_sliced_hat)
                with autocast(enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    loss_mel = F.l1_loss(melspec_sliced, melspec_sliced_hat) * mel_loss_weight
                    loss_kl = kl_loss(z_theta, logs_linspec, mean_text, logs_text, linspec_mask) * kl_loss_weight
                    loss_fm = feature_loss(fmap_real, fmap_gen)
                    loss_gen, losses_gen = generator_loss(disc_gen_hat)
                    loss_final_vae = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            optim_g.zero_grad()
            grad_scaler.scale(loss_final_vae).backward()
            grad_scaler.unscale_(optim_g)
            grad_norm_g = utils.get_grad_values(net_g.parameters(), None)
            grad_scaler.step(optim_g)
            grad_scaler.update()

            if rank == 0:
                if global_step % log_interval == 0:

                    # Text Logging
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                    logger.info("Train Epoch: {} [{:.0f}%]".format(epoch, 100.0 * batch_idx / len(train_dl)))
                    logger.info([grad_norm_d, grad_norm_g] + [x.item() for x in losses] + [global_step, lr])

                    # Tensorboard Scalars
                    scalar_dict = {
                        "grad/grad_norm_d": grad_norm_d,
                        "grad/grad_norm_g": grad_norm_g,
                        "learning_rate": lr,
                        "loss/loss_final_vae": loss_final_vae,
                        "loss/loss_disc": loss_disc,
                        "loss/loss_gen": loss_gen,
                        "loss/loss_feature_map": loss_fm,
                        "loss/loss_mel": loss_mel,
                        "loss/loss_duration": loss_dur,
                        "loss/loss_kl_divergence": loss_kl,
                    }
                    scalar_dict.update({f"loss/loss_gen/{i}": v for i, v in enumerate(losses_gen)})
                    scalar_dict.update({f"loss/loss_disc/real_{i}": v for i, v in enumerate(losses_disc_r)})
                    scalar_dict.update({f"loss/loss_disc/gen_{i}": v for i, v in enumerate(losses_disc_g)})

                    # Tensorboard Images
                    image_dict = {
                        "spec/melspec_sliced_real": utils.plot_spectrogram_to_numpy(
                            melspec_sliced[0].data.cpu().numpy()
                        ),
                        "spec/melspec_sliced_gen": utils.plot_spectrogram_to_numpy(
                            melspec_sliced_hat[0].data.cpu().numpy()
                        ),
                        "attention": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
                    }

                    utils.add_tensorboard_items(
                        writer=writer_train, global_step=global_step, images=image_dict, scalars=scalar_dict
                    )

                if global_step % eval_interval == 0:
                    evaluate(
                        rank,
                        global_step,
                        net_g,
                        eval_dl,
                        writer_eval,
                        hop_length,
                        sampling_rate,
                        n_fft,
                        win_length,
                        n_mels,
                        sampling_rate,
                        fmin,
                        fmax,
                        center,
                    )
                    utils.save_checkpoint(
                        net_g, optim_g, epoch, os.path.join(model_dir_path, f"G_{global_step}.pth"), logger
                    )
                    utils.save_checkpoint(
                        net_d, optim_d, epoch, os.path.join(model_dir_path, f"D_{global_step}.pth"), logger
                    )

            global_step += 1

    for epoch in range(epoch_start, n_epochs + 1):
        train_one_epoch(epoch)
        scheduler_g.step()
        scheduler_d.step()
        if rank == 0:
            logger.info(f"============= Epoch {epoch} ends. =============")


def evaluate(
    rank,
    global_step,
    net_g,
    eval_dl,
    writer_eval,
    hop_length,
    audio_sampling_rate,
    n_fft,
    win_length,
    n_mels,
    sampling_rate,
    fmin,
    fmax,
    center,
):
    net_g.eval()
    with torch.no_grad():
        for batch in eval_dl:
            text, text_lengths, wav, wav_lengths, linspec, linspec_lengths, melspec, melspec_lengths, sid = batch
            text, text_lengths = text.cuda(rank), text_lengths.cuda(rank)
            wav, wav_lengths = wav.cuda(rank), wav_lengths.cuda(rank)
            linspec, linspec_lengths = linspec.cuda(rank), linspec_lengths.cuda(rank)
            melspec, melspec_lengths = melspec.cuda(rank), melspec_lengths.cuda(rank)
            sid = sid.cuda(rank)

            text, text_lengths = text[:1], text_lengths[:1]
            wav, wav_lengths = wav[:1], wav_lengths[:1]
            linspec, linspec_lengths = linspec[:1], linspec_lengths[:1]
            melspec, melspec_lengths = melspec[:1], melspec_lengths[:1]
            sid = sid[:1]
            break

        wav_hat, _, mask, *_ = net_g.module.infer(text, text_lengths, sid, max_len=1000)
        wav_hat_lengths = mask.sum([1, 2]).long() * hop_length
        melspec_hat = utils.audio_utils.wav_to_melspec(
            wav_hat, n_fft, hop_length, win_length, n_mels, sampling_rate, fmin, fmax, center
        )

        image_dict = {"spec/melspec_gen": utils.plot_spectrogram_to_numpy(melspec_hat[0].cpu().numpy())}
        audio_dict = {"wav/wav_gen": wav_hat[0, :, : wav_hat_lengths[0]]}
        if global_step == 0:
            image_dict.update({"spec/melspec_real": utils.plot_spectrogram_to_numpy(melspec[0].cpu().numpy())})
            audio_dict.update({"wav/wav_real": wav[0, :, : wav_lengths[0]]})

        utils.add_tensorboard_items(
            writer=writer_eval,
            global_step=global_step,
            images=image_dict,
            audios=audio_dict,
            audio_sampling_rate=audio_sampling_rate,
        )

        net_g.train()


if __name__ == "__main__":
    main()
