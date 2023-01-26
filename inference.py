import re
from g2pk import G2p
import torch
import numpy as np
from scipy.io import wavfile
from utils import load_checkpoint, get_hparams
from utils.text_utils import SYMBOLS, PAD, SPACE, PUNC, text_to_sequence, add_blank
from network import SynthesizerTrn


g2p = G2p()

others = "".join(PAD + SPACE + PUNC)
not_in_vocab = re.compile(f"[^가-힣{others}]")


def is_not_in_vocab(text):
    global not_in_vocab
    return bool(not_in_vocab.search(text))


def clean_text(text):
    global g2p
    return g2p(text)


def get_text(text):
    cleaned_text = clean_text(text)
    sequence = text_to_sequence(cleaned_text)
    sequence = add_blank(sequence)
    return torch.LongTensor(sequence)


def get_model(model_filename, hps, device, for_new_sid=False):
    if for_new_sid:
        n_speakers = hps.new_sid.pretrained_n_speakers + hps.new_sid.new_n_speakers
    else:
        n_speakers = hps.new_sid.pretrained_n_speakers

    model = SynthesizerTrn(
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
        n_speakers=n_speakers,
        gin_channels=hps.model.gin_channels,
    ).to(device)
    model, _, _ = load_checkpoint(model_filename, model)
    _ = model.eval()
    return model


def text_to_speech(text, speaker_id, model, device):

    sid = torch.LongTensor([speaker_id]).to(device)
    stn_tst = get_text(text).to(device)

    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0
        ]
        audio = audio.data.float().cpu().numpy()

    return audio


def split_text(text):
    puncs = PUNC.copy()
    if "," in puncs:
        puncs.remove(",")
    if " " in puncs:
        puncs.remove("_")
    for punc in puncs:
        text = text.replace(punc + " ", punc + "|")
    text = text.split("|")
    if type(text) == str:
        return [text]
    return text


def texts_to_speech(texts, speaker_ids, model, device):
    if type(speaker_ids) == int:
        speaker_ids = [speaker_ids] * len(texts)
    elif len(speaker_ids) != len(texts):
        speaker_ids = speaker_ids * (len(texts) // len(speaker_ids) + 1)
        speaker_ids = speaker_ids[: len(texts)]

    audio = []
    for text, speaker_id in zip(texts, speaker_ids):
        audio.append(text_to_speech(text, speaker_id, model, device))
    return np.concatenate(audio)


def save_wav(data, sr, target_filename):
    wavfile.write(target_filename, sr, data)


def parse_args():
    import argparse

    desc = "config_filename and model_filename is necessary."

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--config_filename", type=str, help="config filename including path", default="./configs/inference_demo.json"
    )
    parser.add_argument(
        "--model_filename", type=str, help="model pth filename including path", default="./logs/ver1_56/G_853000.pth"
    )
    parser.add_argument(
        "--model_for_new_sid_filename",
        type=str,
        help="model for new sid pth filename including path",
        default=None,
        # default="./logs/new_sid_with_orbit/G_352000.pth",
    )
    parser.add_argument("--wav_saving_path", type=str, help="wav saving dir path", default="./synthesized/")
    parser.add_argument("--sampling_rate", type=int, help="sampling rate", default=22050)
    parser.add_argument("--gpu_num", type=int, help="GPU number", default=-1)

    return parser.parse_args()


def main(args):
    import os

    model_filename = args.model_filename
    model_for_new_sid_filename = args.model_for_new_sid_filename
    config_filename = args.config_filename
    wav_saving_path = args.wav_saving_path
    sampling_rate = args.sampling_rate
    gpu_num = args.gpu_num

    if gpu_num == -1:
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
        device = "cuda:0"

    hps = get_hparams(config_filename)
    if model_for_new_sid_filename is not None:
        n_speakers = hps.new_sid.pretrained_n_speakers + hps.new_sid.new_n_speakers
    else:
        n_speakers = hps.new_sid.pretrained_n_speakers

    print("Loading model file...")
    tts_model = get_model(model_filename, hps, device)
    if model_for_new_sid_filename is not None:
        tts_model_for_new_sid = get_model(model_for_new_sid_filename, hps, device, for_new_sid=True)

    if not os.path.exists(wav_saving_path):
        os.makedirs(wav_saving_path)

    print("Type `q` in Text to stop loop.")
    sample_number = 1
    while True:

        text = input("Text: ")
        if text == "q":
            break
        elif text == "":
            print("Empty text cannot be synthesized.")
            continue
        elif is_not_in_vocab(text):
            print(f"Text must include only hangul and {SPACE + PUNC}.")
            continue

        speaker_id = input(f"Speaker ID (under {n_speakers}): ")
        try:
            speaker_id = int(speaker_id)
        except Exception:
            if speaker_id == "":
                speaker_id = 0
            else:
                speaker_id = None
        if speaker_id is None or speaker_id >= n_speakers:
            print(f"Speaker ID must be less than {n_speakers}.")
            continue

        print("Synthesizing speech...")
        if speaker_id < hps.new_sid.pretrained_n_speakers:
            speech = texts_to_speech(split_text(text), speaker_id, tts_model, device)
        elif model_for_new_sid_filename is not None:
            speech = texts_to_speech(split_text(text), speaker_id, tts_model_for_new_sid, device)
        print("Saving audio file...")
        save_wav(speech, sampling_rate, os.path.join(wav_saving_path, f"tts_sample_{sample_number}.wav"))
        print("Done!\n")
        sample_number += 1


if __name__ == "__main__":
    args = parse_args()
    main(args)
