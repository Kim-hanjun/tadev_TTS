import pydub
import numpy as np
from scipy.io import wavfile
from typing import Tuple
import os


def load_wav(wavfilename: str) -> Tuple[np.array, int]:
    """load wav file"""
    sr, wav = wavfile.read(wavfilename)
    assert len(wav.shape) == 1, "wav file channel must be 1 (mono)."
    return wav, sr


def save_wav(data: np.array, sr: int, target_filename: str) -> None:
    wavfile.write(target_filename, sr, data)


def resample_wav(wav: np.array, origin_sr: int, target_sr: int) -> np.array:
    data = pydub.AudioSegment(wav.tobytes(), frame_rate=origin_sr, sample_width=wav.dtype.itemsize, channels=1)
    data = data.set_frame_rate(target_sr)
    resampled_wav = np.frombuffer(data._data, dtype="int16")
    return resampled_wav


base_foldername = "/home/jun/workspace/m4a_to_wav/주호 녹음 파일"
wav_foldername = "/home/jun/workspace/m4a_to_wav/주호녹음wav1"
for dirpath, dirnames, filenames in os.walk(base_foldername):
    for m4a_filename in filenames:
        full_filename = os.path.join(dirpath, m4a_filename)
        sound = pydub.AudioSegment.from_file(full_filename, format="m4a")
        sound = sound.set_channels(1)
        wav_filename = os.path.splitext(m4a_filename)[0]
        file_handle = sound.export(os.path.join(wav_foldername, wav_filename + ".wav"), format="wav")
        save_filename = os.path.join(wav_foldername, wav_filename + ".wav")
        wav, origin_sr = load_wav(save_filename)
        wav = resample_wav(wav, origin_sr, 16000)
        save_wav(wav, 16000, save_filename)
