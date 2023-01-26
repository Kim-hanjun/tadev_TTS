from scipy.io import wavfile
import os
from pydub import AudioSegment
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavfile_path", type=str, default="/home/jun/workspace/sampling/0004_G2A2E7_PNR_000001.wav")
    parser.add_argument("--destination_path", type=str, default="/home/jun/workspace/tadev-TTS/test.wav")
    return parser.parse_args()


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    """
    sound is a pydub.AudioSegment
    ex) sound = AudioSegment.from_file("/home/jun/workspace/sampling/0004_G2A2E7_PNR_000001.wav", format="wav")

    silence_threshold in dB
    chunk_size in ms
    iterate over chunks until you find the first one with sound
    """
    trim_ms = 0  # ms
    while sound[trim_ms : trim_ms + chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms


def main(args):
    sound = AudioSegment.from_file(args.wavfile_path, format="wav")

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    trimmed_sound = sound[start_trim : duration - end_trim]
    trimmed_sound.export(args.destination_path, format="wav")


if __name__ == "__main__":
    args = get_args()
    main(args)
