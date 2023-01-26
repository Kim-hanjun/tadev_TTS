import os
from tqdm import tqdm
import torch
from .audio_utils import get_wav_linspec_melspec
from .text_utils import text_to_sequence, add_blank
from typing import List

import torch.distributed as dist
import torch.multiprocessing as mp


def load_wavfilename_sid_text(filename: str, split_symbol: str = "|") -> List[List[str]]:
    """clean이 완료된 cleaned file을 load"""
    with open(filename, "r") as f:
        filepath_sid_text = [line.strip().split(split_symbol) for line in f]
    return filepath_sid_text


sid_dict = {}


class TextAudioSpeakerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filepath_sid_text_filename: str,
        sr: int,
        max_wav_value: float,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        fmin: int,
        fmax: int,
        min_text_len: int = None,
        max_text_len: int = None,
        add_blank: bool = True,
        start_sid: int = 0,
    ) -> None:

        """wav, linspec, melspec, sid, text를 return하는 dataset

        Args:
            filepath_sid_text_filename (str): cleaned file name
            sr (int): sampling rate
            max_wav_value (float): wav의 절대값의 max value
            n_fft (int): n fft
            hop_length (int): window사이의 간격
            win_length (int): window의 길이
            n_mels (int): n mels
            fmin (int): fmin
            fmax (int): fmax
            min_text_len (int): 본 값보다 짧은 text는 filtering
            max_text_len (int): 본 값보다 긴 text는 filtering
            add_blank (bool): 텍스트 사이에 blank를 넣을 것인지 여부
        """

        self.filepath_sid_text = load_wavfilename_sid_text(filepath_sid_text_filename)
        self.max_wav_value = max_wav_value
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.add_blank = add_blank
        self.start_sid = start_sid
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        if self.min_text_len is None:
            self.min_text_len = 1
        if self.max_text_len is None:
            self.max_text_len = 190

        self._filter_data()

    def _filter_data(self):
        """sid 0부터 순서대로 mapping, 너무 길거나 짧은 데이터는 제외, self.length는 bucket sampler에서 사용된다."""
        audiopaths_sid_text_filtered = []
        lengths = []
        global sid_dict
        sid_num = self.start_sid
        for audio_path, sid, text in self.filepath_sid_text:
            if sid not in sid_dict.keys():
                sid_dict[sid] = sid_num
                sid_num += 1
            sid = sid_dict[sid]
            text = text_to_sequence(text)
            if len(text) <= self.min_text_len or len(text) >= self.max_text_len:
                continue
            audiopaths_sid_text_filtered.append([audio_path, sid, text])
            lengths.append(os.path.getsize(audio_path) // (2 * self.hop_length))
        print(f"Filtering result: data length {len(self.filepath_sid_text)} -> {len(audiopaths_sid_text_filtered)}")
        self.filepath_sid_text = audiopaths_sid_text_filtered
        self.lengths = lengths
        self.sid_dict = sid_dict

    def get_text(self, text: str):
        """옵션에 따라 각 토큰사이에 공백을 추가한다."""
        if self.add_blank:
            text = add_blank(text)
        text = torch.LongTensor(text)
        return text

    def get_audio(self, wavfilename: str):
        """wav, linear spectrogram, mel spectrogram을 load"""
        wav_norm, linspec, melspec = get_wav_linspec_melspec(
            wavfilename,
            self.max_wav_value,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.n_mels,
            self.fmin,
            self.fmax,
            self.sr,
        )
        return wav_norm, linspec, melspec

    def get_sid(self, sid: str):
        sid = torch.LongTensor([sid])
        return sid

    def get_text_audio_sid(self, wavfilename: str, sid: str, text: str):
        text = self.get_text(text)
        wav, linspec, melspec = self.get_audio(wavfilename)
        sid = self.get_sid(sid)
        return text, wav, linspec, melspec, sid

    def __getitem__(self, idx):
        return self.get_text_audio_sid(*self.filepath_sid_text[idx])

    def __len__(self):
        return len(self.filepath_sid_text)


class TextAudioSpeakerCollate:
    def __init__(self, return_ids: bool = False):
        """Class for calling collate function

        batch는 [text, wav, linspec, melspec, sid] 로 이루어져있다.
        각 batch마다 linear spectrogram의 길이에 따라 정렬한다.
        text, wav, linspec, melspec는 최대길이를 가지는 데이터에 맞게 pad 수행.
        """
        self.return_ids = return_ids

    def __call__(self, batch):
        # Sort by linspec
        _, ids_sorted_decreasing = torch.sort(torch.LongTensor([x[2].size(1) for x in batch]), dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        text_padded = torch.LongTensor(len(batch), max_text_len)
        text_padded.zero_()
        text_lengths = torch.LongTensor(len(batch))

        max_wav_len = max([x[1].size(1) for x in batch])
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        wav_padded.zero_()
        wav_lengths = torch.LongTensor(len(batch))

        max_linspec_len = max([x[2].size(1) for x in batch])
        linspec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_linspec_len)
        linspec_padded.zero_()
        linspec_lengths = torch.LongTensor(len(batch))

        max_melspec_len = max([x[3].size(1) for x in batch])
        melspec_padded = torch.FloatTensor(len(batch), batch[0][3].size(0), max_melspec_len)
        melspec_padded.zero_()
        melspec_lengths = torch.LongTensor(len(batch))

        sid = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, : text.size(0)] = text
            text_lengths[i] = text.size(0)

            wav = row[1]
            wav_padded[i, :, : wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            linspec = row[2]
            linspec_padded[i, :, : linspec.size(1)] = linspec
            linspec_lengths[i] = linspec.size(1)

            melspec = row[3]
            melspec_padded[i, :, : melspec.size(1)] = melspec
            melspec_lengths[i] = melspec.size(1)

            sid[i] = row[4]

        _returns = (
            text_padded,
            text_lengths,
            wav_padded,
            wav_lengths,
            linspec_padded,
            linspec_lengths,
            melspec_padded,
            melspec_lengths,
            sid,
        )
        if self.return_ids:
            _returns = _returns + (ids_sorted_decreasing,)

        return _returns


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)  # data의 index를 해당 bucket에 append
            else:
                print(f"Data is out of boundaries. data length: {length}")

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)  # data가 없는 bucket과 boundary는 pop

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size  # total_batch_size = n_gpus * batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)  # # ?
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[: (rem % len_bucket)]

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size : (j + 1) * self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        """data length를 input으로 받아 해당 data가 어떤 boundary에 속하는지 return"""
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


def main_worker(rank, n_gpus):

    CLEANED_FILENAME = "./dummies/filelist/ver1_train_cleaned.txt"
    SAMPLING_RATE = 22050
    MAX_WAV_VALUE = 32768.0
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    N_MELS = 80
    FMIN = 0
    FMAX = None
    MIN_TEXT_LEN = 1
    MAX_TEXT_LEN = 190
    BATCH_SIZE = 8
    NUM_WORKERS = 8

    dist.init_process_group(backend="nccl", init_method="env://", world_size=n_gpus, rank=rank)
    torch.manual_seed(1234)
    torch.cuda.set_device(rank)

    ds = TextAudioSpeakerDataset(
        CLEANED_FILENAME,
        SAMPLING_RATE,
        MAX_WAV_VALUE,
        N_FFT,
        HOP_LENGTH,
        WIN_LENGTH,
        N_MELS,
        FMIN,
        FMAX,
        MIN_TEXT_LEN,
        MAX_TEXT_LEN,
    )
    collate_fn = TextAudioSpeakerCollate()
    n_gpus = torch.cuda.device_count()
    sampler = DistributedBucketSampler(
        ds, BATCH_SIZE, [32, 300, 400, 500, 600, 700, 800, 900, 1000], num_replicas=n_gpus, rank=None, shuffle=True
    )
    dl = torch.utils.data.DataLoader(
        ds, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True, collate_fn=collate_fn, batch_sampler=sampler
    )
    pbar = tqdm(dl)
    for i, batch in enumerate(pbar):
        (
            text_padded,
            text_lengths,
            wav_padded,
            wav_lengths,
            linspec_padded,
            linspec_lengths,
            melspec_padded,
            melspec_lengths,
            sid,
        ) = batch
        pass
        # text_padded: [batch_size, text_len]
        # text_lengths: [batch_size]
        # wav_padded: [batch_size, 1, wav_len]
        # wav_lengths: [batch_size]
        # linspec_padded: [batch_size, n_linspec_channels, linspec_len(=melspec_len)]
        # linspec_lengths: [batch_size]
        # melspec_padded: [batch_size, n_melspec_channels, melspec_len(=linspec_len)]
        # melspec_lengths: [batch_size]
        # sid: [batch_size]


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8899"

    mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus,))


if __name__ == "__main__":

    main()
