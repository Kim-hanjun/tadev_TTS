from utils.audio_utils import ffmpeg_normalize, load_wav, save_wav, resample_wav, auto_vad, add_blank_to_wav
from pydub import AudioSegment
import argparse
import numpy as np
import os
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavfile_dir", type=str, default="/home/jun/workspace/sampling/sampled/VCTK/wav")
    parser.add_argument(
        "--destination_dir", type=str, default="/home/jun/workspace/sampling/sampled/VCTK/preprocessed_wav"
    )
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--frame_duration_ms", type=int, default=10)
    parser.add_argument("--set_mode_number", type=int, default=3)
    parser.add_argument("--blank_len", type=float, default=0.3)
    parser.add_argument("--duration_thershold", type=float, default=2)
    return parser.parse_args()


def main(args):
    blank_len = args.blank_len
    target_sr = args.target_sr
    frame_duration_ms = args.frame_duration_ms
    set_mode_number = args.set_mode_number
    duration_thershold = args.duration_thershold
    for dirpath, dirnames, filenames in tqdm(os.walk(args.wavfile_dir)):
        print(dirpath)
        for wav_filename in tqdm(filenames):
            full_filename = os.path.join(dirpath, wav_filename)
            wav, origin_sr = load_wav(full_filename)
            wav = resample_wav(wav, origin_sr, target_sr)
            cutted_samples = auto_vad(wav, target_sr, frame_duration_ms, set_mode_number, duration_thershold)
            processed_wav = add_blank_to_wav(cutted_samples, target_sr, blank_len, dtype=np.int16)
            save_dirname = dirpath.replace(args.wavfile_dir, args.destination_dir)
            save_filename = os.path.join(save_dirname, wav_filename)
            os.makedirs(save_dirname, exist_ok=True)
            save_wav(processed_wav, target_sr, save_filename)
            ffmpeg_normalize(save_filename)


if __name__ == "__main__":
    args = get_args()
    main(args)
