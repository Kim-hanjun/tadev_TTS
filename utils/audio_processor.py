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


class AudioProcessor:
    def __init__(
        self,
        sample_rate: int,
        max_wav_value: int,
        blank_len: float = 0.3,
        set_mode_number: int = 3,
        n_fft: int = 1024,
    ):
        self.blank_len = blank_len
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(set_mode_number)
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.max_wav_value = max_wav_value

    def add_blank_to_wav(self, wav: np.array, sr: int = 16000, dtype=np.int16) -> np.array:
        blank = np.zeros(int(self.blank_len * sr), dtype=np.int16)
        blanked_wav = np.concatenate([blank, wav, blank], axis=0)
        return blanked_wav

    def frame_generator(self, frame_duration_ms: int, audio: np.array) -> list:
        frames = []
        assert self.sample_rate in [
            8000,
            16000,
            32000,
            48000,
        ], "sample_rate must be either 8000 or 16000 or 32000 or 48000"
        assert frame_duration_ms in [10, 20, 30], "frame_duration_ms must be either 10 or 20 or 30 "
        # https://github.com/wiseman/py-webrtcvad/issues/30
        chunk_size = int(self.sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        while offset + chunk_size < len(audio):
            frames.append(audio[offset : offset + chunk_size])
            offset += chunk_size

        return frames

    def auto_vad(
        self,
        samples: np.array,
        frame_duration_ms: int,
        duration_threshold: int,
    ) -> np.array:
        """
        frame_duration_ms: must be either 10 or 20 or 30
        set_mode_number: Optionally, set its aggressiveness mode, which is an integer between 0 and 3.
                        0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
        sample_rate: The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.
        duration_threshold: ???????????? ????????? not_speech ????????? ????????? ?????? (???????????? 2)
        """
        frames = self.frame_generator(frame_duration_ms, samples, self.sample_rate)
        chunk_size = int(self.sample_rate * (frame_duration_ms / 1000.0) * 2)
        now = 0
        cutted_samples = np.array([], dtype=np.int16)
        for idx, frame in enumerate(frames):
            if self.vad.is_speech(frame, self.sample_rate):
                continue
            else:
                if idx - now >= duration_threshold:
                    cutted_samples = np.append(cutted_samples, samples[now * chunk_size : idx * chunk_size])
                now = idx + 1
        # for?????? ????????? ???????????? else?????? ????????? ???????????? ????????? ????????? ?????????????????? ????????? append ????????? ????????????
        else:
            cutted_samples = np.append(cutted_samples, samples[now * chunk_size : (idx + 1) * chunk_size])

        return cutted_samples

    # ?????? ???????????? ??????????????? ??? auto_vad ????????? ?????? ???????????? ????????? ?????? ???????????? ??????
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

    def ffmpeg_normalize(self, wavfile_path: str) -> None:
        """
        ?????? ??????(wavfile_path)??? ?????? ????????? ?????? ???????????? ????????? ??????????????? ?????? ????????? ?????? ??????????????? ?????? ??????
        """
        os.system("ffmpeg-normalize " + wavfile_path + " -nt rms -t=-27 -o " + wavfile_path + " -ar 16000 -f")

    @staticmethod
    def load_wav(wavfilename: str) -> Tuple[np.array, int]:
        """load wav file"""
        sr, wav = wavfile.read(wavfilename)
        assert len(wav.shape) == 1, "wav file channel must be 1 (mono)."
        return wav, sr

    @staticmethod
    def save_wav(data: np.array, sr: int, target_filename: str) -> None:
        wavfile.write(target_filename, sr, data)

    @staticmethod
    def resample_wav(wav: np.array, origin_sr: int, target_sr: int) -> np.array:
        data = pydub.AudioSegment(wav.tobytes(), frame_rate=origin_sr, sample_width=wav.dtype.itemsize, channels=1)
        data = data.set_frame_rate(target_sr)
        resampled_wav = np.frombuffer(data._data, dtype="int16")
        return resampled_wav

    def devide_by_max_wav_value(wav: torch.FloatTensor, max_wav_value: int) -> torch.FloatTensor:
        """wav scaling??? ??????"""
        return wav / max_wav_value

    def dynamic_range_compression_torch(self, x, C=1, clip_val=1e-5):
        """
        PARAMS
        ------
        C: compression factor
        """
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def spectral_normalize_torch(self, magnitudes):
        output = self.dynamic_range_compression_torch(magnitudes)
        return output

    mel_basis = {}
    hann_window = {}

    def wav_to_linspec(
        wav: torch.FloatTensor, n_fft: int, hop_length: int, win_length: int, center: bool = False
    ) -> torch.FloatTensor:
        """wav??? linear spectrogram?????? ??????"""
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

        if (
            len_wav_size == 3
        ):  # necessary if wav has shape [b, 1, wav_len]. torch.stft support only for 1D or 2D tensor.
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
        self, linspec: torch.FloatTensor, n_fft: int, n_mels: int, sampling_rate: int, fmin: int, fmax: int
    ) -> torch.FloatTensor:
        """linear spectrogram??? mel spectrogram?????? ??????"""
        global mel_basis
        linspec_dtype = linspec.dtype
        linspec_device = linspec.device
        dtype_device = str(linspec_dtype) + "_" + str(linspec_device)
        fmax_dtype_device = str(fmax) + "_" + dtype_device
        if fmax_dtype_device not in mel_basis:
            mel = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=linspec_dtype, device=linspec_device)
        melspec = torch.matmul(mel_basis[fmax_dtype_device], linspec)
        melspec = self.spectral_normalize_torch(melspec)
        return melspec

    def wav_to_melspec(
        self,
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
        linspec = self.wav_to_linspec(wav, n_fft, hop_length, win_length, center)
        melspec = self.linspec_to_melspec(linspec, n_fft, n_mels, sampling_rate, fmin, fmax)
        return melspec

    def get_wav_linspec_melspec(
        self,
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
        """wavfilename??? ???????????? wav??? sacling?????? linear spectrogram??? mel spectrogram??? ??????"""

        wav, origin_sr = self.load_wav(wavfilename)
        if origin_sr != target_sr:
            wav = self.resample_wav(wav, origin_sr, target_sr)
            self.save_wav(wav, target_sr, wavfilename)
        wav = to_float_tensor(wav)
        wav = self.devide_by_max_wav_value(wav, max_wav_value)

        linspec_filename = wavfilename.replace(".wav", "_linspec.pt")

        if os.path.exists(linspec_filename):
            linspec = torch.load(linspec_filename)
        else:
            linspec = self.wav_to_linspec(wav, n_fft, hop_length, win_length, center)
            torch.save(linspec, linspec_filename)

        melspec = self.linspec_to_melspec(linspec, n_fft, n_mels, target_sr, fmin, fmax)

        wav = wav.unsqueeze(0)

        return wav, linspec, melspec
