from scipy.io import wavfile
import webrtcvad
import pydub
import librosa
import torch
import torch.nn.functional as F
import numpy as np
import os
from .__init__ import to_float_tensor
from typing import Tuple


def add_blank_to_wav(wav: np.array, sr: int = 16000, blank_len: float = 0.3, dtype=np.int16) -> np.array:
    blank = np.zeros(int(blank_len * sr), dtype=np.int16)
    blanked_wav = np.concatenate([blank, wav, blank], axis=0)
    return blanked_wav


def frame_generator(frame_duration_ms: int, audio: np.array, sample_rate: int) -> list:
    frames = []
    assert sample_rate in [8000, 16000, 32000, 48000], "sample_rate must be either 8000 or 16000 or 32000 or 48000"
    assert frame_duration_ms in [10, 20, 30], "frame_duration_ms must be either 10 or 20 or 30 "
    # https://github.com/wiseman/py-webrtcvad/issues/30
    chunk_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + chunk_size < len(audio):
        frames.append(audio[offset : offset + chunk_size])
        offset += chunk_size

    return frames


def auto_vad(
    samples: np.array, sample_rate: int, frame_duration_ms: int, set_mode_number: int, duration_threshold: int
) -> np.array:
    """
    frame_duration_ms: must be either 10 or 20 or 30
    set_mode_number: Optionally, set its aggressiveness mode, which is an integer between 0 and 3.
                    0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
    sample_rate: The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.
    duration_threshold: 음성으로 판단할 not_speech 리스트 값들의 차이 (권고값은 2)
    """
    vad = webrtcvad.Vad()
    vad.set_mode(set_mode_number)
    frames = frame_generator(frame_duration_ms, samples, sample_rate)
    chunk_size = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    now = 0
    cutted_samples = np.array([], dtype=np.int16)
    for idx, frame in enumerate(frames):
        if vad.is_speech(frame, sample_rate):
            continue
        else:
            if idx - now >= duration_threshold:
                cutted_samples = np.append(cutted_samples, samples[now * chunk_size : idx * chunk_size])
            now = idx + 1
    # for문이 끝나고 실행되는 else문은 음성의 마지막이 묵음이 아닐때 마지막까지의 음성을 append 해주기 위함이다
    else:
        cutted_samples = np.append(cutted_samples, samples[now * chunk_size : (idx + 1) * chunk_size])

    return cutted_samples


# 묵음 제거하는 함수이지만 위 auto_vad 함수를 주로 사용하기 때문에 추후 삭제해도 무방
def detect_leading_silence(sound: np.array, silence_threshold=-50.0, chunk_size=10) -> int:
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


def ffmpeg_normalize(wavfile_path: str) -> None:
    """
    현재 파일(wavfile_path)에 바로 적용이 되는 기능이니 백업이 가능하도록 원본 파일을 미리 마련해두고 사용 요망
    """
    os.system("ffmpeg-normalize " + wavfile_path + " -nt rms -t=-27 -o " + wavfile_path + " -ar 16000 -f")


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


def devide_by_max_wav_value(wav: torch.FloatTensor, max_wav_value: int) -> torch.FloatTensor:
    """wav scaling에 사용"""
    return wav / max_wav_value


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def wav_to_linspec(
    wav: torch.FloatTensor, n_fft: int, hop_length: int, win_length: int, center: bool = False
) -> torch.FloatTensor:
    """wav를 linear spectrogram으로 만듦"""
    if torch.min(wav) < -1.0:
        print("min value is ", torch.min(wav))
    if torch.max(wav) > 1.0:
        print("max value is ", torch.max(wav))

    global hann_window
    wav_dtype = wav.dtype
    wav_device = wav.device
    dtype_device = str(wav_dtype) + "_" + str(wav_device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=wav_dtype, device=wav_device)

    pad_len = int((n_fft - hop_length) / 2)
    len_wav_size = len(wav.size())
    if len_wav_size == 1:  # necessary if torch version doesn't support 1D and 2D padding with non-constant padding
        wav = wav.unsqueeze(0).unsqueeze(0)
    wav = F.pad(wav, (pad_len, pad_len), mode="reflect")
    if len_wav_size == 1:
        wav = wav.squeeze(0).squeeze(0)

    if len_wav_size == 3:  # necessary if wav has shape [b, 1, wav_len]. torch.stft support only for 1D or 2D tensor.
        wav = wav.squeeze(1)
    linspec = torch.stft(
        wav,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        # center=center, pad_mode="reflect", normalized=False, onesided=True, return_complex=False
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )
    linspec = torch.sqrt(linspec.pow(2).sum(-1) + 1e-6)
    return linspec


def linspec_to_melspec(
    linspec: torch.FloatTensor, n_fft: int, n_mels: int, sampling_rate: int, fmin: int, fmax: int
) -> torch.FloatTensor:
    """linear spectrogram을 mel spectrogram으로 만듦"""
    global mel_basis
    linspec_dtype = linspec.dtype
    linspec_device = linspec.device
    dtype_device = str(linspec_dtype) + "_" + str(linspec_device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=linspec_dtype, device=linspec_device)
    melspec = torch.matmul(mel_basis[fmax_dtype_device], linspec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def wav_to_melspec(
    wav: torch.FloatTensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    sampling_rate: int,
    fmin: int,
    fmax: int,
    center: bool = False,
) -> torch.FloatTensor:
    linspec = wav_to_linspec(wav, n_fft, hop_length, win_length, center)
    melspec = linspec_to_melspec(linspec, n_fft, n_mels, sampling_rate, fmin, fmax)
    return melspec


def get_wav_linspec_melspec(
    wavfilename: str,
    max_wav_value: float,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: int,
    fmax: int,
    target_sr: int = None,
    center: bool = False,
) -> Tuple[torch.FloatTensor]:
    """wavfilename을 불러와서 wav를 sacling하고 linear spectrogram과 mel spectrogram을 만듦"""

    wav, origin_sr = load_wav(wavfilename)
    if origin_sr != target_sr:
        wav = resample_wav(wav, origin_sr, target_sr)
        save_wav(wav, target_sr, wavfilename)
    wav = to_float_tensor(wav)
    wav = devide_by_max_wav_value(wav, max_wav_value)

    linspec_filename = wavfilename.replace(".wav", "_linspec.pt")

    if os.path.exists(linspec_filename):
        linspec = torch.load(linspec_filename)
    else:
        linspec = wav_to_linspec(wav, n_fft, hop_length, win_length, center)
        torch.save(linspec, linspec_filename)

    melspec = linspec_to_melspec(linspec, n_fft, n_mels, target_sr, fmin, fmax)

    wav = wav.unsqueeze(0)

    return wav, linspec, melspec
