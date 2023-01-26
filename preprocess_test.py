from utils.audio_utils import ffmpeg_normalize, load_wav, save_wav, resample_wav, auto_vad, add_blank_to_wav
from pydub import AudioSegment
import argparse
import numpy as np
import os
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wavfile_dir", type=str, default="/home/jun/workspace/sampling/sampled/AIHUB/json/7533_G1A2E7_LTG"
    )
    parser.add_argument(
        "--destination_dir", type=str, default="/home/jun/workspace/sampling/sampled/AIHUB/preprocessed_wav"
    )
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--frame_duration_ms", type=int, default=10)
    parser.add_argument("--set_mode_number", type=int, default=3)
    parser.add_argument("--blank_len", type=float, default=0.3)
    parser.add_argument("--judgment_voice", type=int, default=2)
    return parser.parse_args()


def main(args):
    json_list = os.listdir(args.wavfile_dir)
    filenames1 = []
    for json in json_list:
        json1 = json.replace(".json", "")
        filenames1.append(json1)
    wav_list = os.listdir("/home/jun/workspace/sampling/sampled/AIHUB/preprocessed_wav/7533_G1A2E7_LTG")
    filenames2 = []
    for wav in wav_list:
        wav1 = wav.replace(".wav", "")
        filenames2.append(wav1)
    for wav_filename in filenames2:
        if wav_filename not in filenames1:
            remove_wav = os.path.join(
                "/home/jun/workspace/sampling/sampled/AIHUB/preprocessed_wav/7533_G1A2E7_LTG", wav_filename + ".wav"
            )
            os.remove(remove_wav)


if __name__ == "__main__":
    args = get_args()
    main(args)
