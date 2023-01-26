from utils.audio_utils import ffmpeg_normalize, load_wav, save_wav, resample_wav, auto_vad, add_blank_to_wav
from pydub import AudioSegment
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavfile_path", type=str, default="/home/jun/workspace/sampling/sampled/VCTK/wav/p259/p259_254.wav"
    )
    parser.add_argument("--destination_path", type=str, default="/home/jun/workspace/tadev-TTS/eng.wav")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--frame_duration_ms", type=int, default=10)
    parser.add_argument("--set_mode_number", type=int, default=3)
    parser.add_argument("--blank_len", type=float, default=0.3)
    return parser.parse_args()


def main(args):
    target_sr = args.target_sr
    frame_duration_ms = args.frame_duration_ms
    set_mode_number = args.set_mode_number
    blank_len = args.blank_len
    wav, origin_sr = load_wav(args.wavfile_path)
    wav = resample_wav(wav, origin_sr, target_sr)
    cutted_samples = auto_vad(wav, target_sr, frame_duration_ms, set_mode_number, 2)
    processed_wav = add_blank_to_wav(cutted_samples, target_sr, blank_len, dtype=np.int16)
    save_wav(processed_wav, target_sr, args.destination_path)
    ffmpeg_normalize(args.destination_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
