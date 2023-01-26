import webrtcvad
import numpy as np
from scipy.io import wavfile
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavfile_path", type=str, default="/home/jun/workspace/sampling/resampled.wav")
    parser.add_argument("--destination_path", type=str, default="/home/jun/workspace/tadev-TTS/test.wav")
    return parser.parse_args()


def frame_generator(frame_duration_ms: int, audio: np.array, sample_rate: int):
    frames = []
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 320 or 640 or 960
    offset = 0
    while offset + n < len(audio):
        frames.append(audio[offset : offset + n])
        offset += n

    return frames


def auto_vad(samples: np.array, sample_rate: int, frame_duration_ms: int, set_mode_number: int):
    """
    frame_duration_ms: must be either 10 or 20 or 30
    set_mode_number: Optionally, set its aggressiveness mode, which is an integer between 0 and 3.
                    0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
    sample_rate: The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.
    """
    vad = webrtcvad.Vad()
    vad.set_mode(set_mode_number)
    not_speech = []
    frames = frame_generator(frame_duration_ms, samples, sample_rate)
    for idx, frame in enumerate(frames):
        if not vad.is_speech(frame, sample_rate):
            not_speech.append(idx)

    prior = 0
    cutted_samples = []
    print(samples.size)
    n_frame = len(frames)
    for i in not_speech:
        if i - prior > 2:
            start = int((float(prior) * n_frame))
            end = int((float(i) * n_frame))
            # start = int((float(prior) * int(sample_rate * (frame_duration_ms / 1000.0) * 2)))
            # end = int((float(i) * int(sample_rate * (frame_duration_ms / 1000.0) * 2)))
            print(start, end)
            if len(cutted_samples) == 0:
                cutted_samples = samples[start:end]
            else:
                cutted_samples = np.append(cutted_samples, samples[start:end])
        prior = i
    return cutted_samples


def main(args):
    sample_rate, samples = wavfile.read(args.wavfile_path)
    frame_duration_ms = 10
    set_mode_number = 3
    cutted_samples = auto_vad(samples, sample_rate, frame_duration_ms, set_mode_number)
    wavfile.write(args.destination_path, sample_rate, cutted_samples)


if __name__ == "__main__":
    args = get_args()
    main(args)
