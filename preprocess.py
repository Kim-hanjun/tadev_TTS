import os
from tqdm import tqdm
import json
import torch

# from g2pk import G2p
from typing import Dict, Union
from utils import get_logger, to_float_tensor
from utils.audio_utils import load_wav, resample_wav, save_wav, devide_by_max_wav_value, wav_to_linspec


# g2p = G2p()


def check_json(json_data: Dict[str, Union[str, float]], dir_number_of_speaker: str, dir_speaker_name: str) -> str:

    """AiHubMultiSpeaker 데이터의 json파일을 통해 cleansing 진행

    기본정보
        Language - KOR 아니면 처리.
        NumberOfSpeaker - SpeakerID 구분. 폴더로 구분된 것과 다르면 처리.
    음성정보
        SamplingRate - 48000 말고 다른것도 있나 확인. 있으면 처리.
        NumberOfBit - SamplingRate와 마찬가지로 16이 아니면 처리.
        NumberOfChannel - 마찬가지로 1이 아니면 처리.
    화자정보
        SpeakerName - 폴더로 구분된 것과 다르면 처리.
    파일정보
        FileCategory - Audio 아니면 처리
        FileName - 해당 .wav 파일이 없으면 처리.
        FileFormat - WAV 아니면 처리.
        NumberOfRepeat - 1 아닌거 찾아보기.
    기타정보
        QualityStatus - Good 아니면 처리.

    Args:
        json_data: open된 json data
        dir_number_of_speaker (str): dir에 써있는 number_of_speaker(speaker id)
        dir_speaker_name (str): dir에 써있는 speaker_name(speaker 이니셜)

    Returns:
        Cleansing 결과를 return
        데이터가 정상이라면 ""을 return 한다.
    """

    full_info = ""

    text_info = "전사정보\n"
    text_info += f"\tOrgLabelText    : {json_data['전사정보']['OrgLabelText']}\n"
    text_info += f"\tTransLabelText  : {json_data['전사정보']['TransLabelText']}\n"
    text_info += f"\tRefinedLabelText: {json_data['전사정보']['RefinedLabelText']}\n"

    # 텍스트 파일 자체가 없는 경우
    if (not json_data["전사정보"]["OrgLabelText"]) or (not json_data["전사정보"]["TransLabelText"]):
        return text_info

    basic_info = ""
    if json_data["기본정보"]["Language"] != "KOR":
        basic_info += f"\tLanguage: {json_data['기본정보']['Language']}\n"
    if json_data["기본정보"]["NumberOfSpeaker"] != dir_number_of_speaker:
        basic_info += (
            f"\tNumberOfSpeaker: {json_data['기본정보']['NumberOfSpeaker']}, DirNumberOfSpeaker: {dir_number_of_speaker}\n"
        )
    if basic_info:
        basic_info = "기본정보:\n" + basic_info
    full_info += basic_info

    speech_info = ""
    # if json_data["음성정보"]["SamplingRate"] != "48000":
    #     speech_info += f"\tSamplingRate: {json_data['음성정보']['SamplingRate']}\n"
    if json_data["음성정보"]["NumberOfBit"] != "16":
        speech_info += f"\tNumberOfBit: {json_data['음성정보']['NumberOfBit']}\n"
    if json_data["음성정보"]["NumberOfChannel"] != "1":
        speech_info += f"\tNumberOfChannel: {json_data['음성정보']['NumberOfChannel']}\n"
    if speech_info:
        speech_info = "음성정보:\n" + speech_info
    full_info += speech_info

    speaker_info = ""
    if json_data["화자정보"]["SpeakerName"] != dir_speaker_name:
        speaker_info += f"\tSpeakerName: {json_data['화자정보']['SpeakerName']}, DirSpeakerName: {dir_speaker_name}\n"
    if speaker_info:
        speaker_info = "화자정보:\n" + speaker_info
    full_info += speaker_info

    file_info = ""
    if json_data["파일정보"]["FileCategory"] != "Audio":
        file_info += f"\tFileCategory: {json_data['파일정보']['FileCategory']}\n"
    if json_data["파일정보"]["FileFormat"] != "WAV":
        file_info += f"\tFileFormat: {json_data['파일정보']['FileFormat']}\n"
    if json_data["파일정보"]["NumberOfRepeat"] != "1":
        file_info += f"\tNumberOfRepeat: {json_data['파일정보']['NumberOfRepeat']}\n"
    if file_info:
        file_info = "파일정보:\n" + file_info
    full_info += file_info

    etc_info = ""
    if json_data["기타정보"]["QualityStatus"] != "Good":
        etc_info += f"\tQualityStatus: {json_data['기타정보']['QualityStatus']}\n"
    if etc_info:
        etc_info = "기타정보:\n" + etc_info
    full_info += etc_info

    if full_info:
        full_info = text_info + full_info

    return full_info


def preprocess_one_speaker(json_path: str, wav_path: str, log_cleansing: bool = False, logger=None) -> None:

    # global g2p

    json_file_list = os.listdir(json_path)
    speaker_id, _, speaker_name = os.path.split(json_path)[-1].split("_")

    pbar = tqdm(json_file_list)
    one_speaker_cleaned_info = ""
    for json_filename in pbar:
        json_filename = os.path.join(json_path, json_filename)

        with open(json_filename) as f:
            json_data = json.load(f)

        # wavfilename
        wav_filename = os.path.join(wav_path, json_data["파일정보"]["FileName"])
        if not os.path.exists(wav_filename):  # wavfile 자체가 없는 경우
            if log_cleansing:
                logger.info(f"wav file does not exist: {wav_filename}")
            continue

        full_info = check_json(json_data, speaker_id, speaker_name)
        if full_info:  # full_info에 값이 있는 경우(데이터가 잘못된 경우)
            if log_cleansing:
                logger.info(full_info)
                full_info = json_filename + "\n" + full_info
            continue

        # speaker id
        sid = json_data["기본정보"]["NumberOfSpeaker"]

        # text
        if json_data["전사정보"]["RefinedLabelText"]:
            text = json_data["전사정보"]["RefinedLabelText"]
        else:
            text = json_data["전사정보"]["TransLabelText"]

        # cleaned_text = g2p(text)
        cleaned_info = "|".join([wav_filename, sid, text])

        one_speaker_cleaned_info += cleaned_info + "\n"

    return one_speaker_cleaned_info


def save_resampled_wav_linspec(
    origin_wav_filename: str,
    target_wav_filename: str,
    target_sr: int,
    max_wav_value: Union[int, float],
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> None:

    # Resample wav & Save resampled wav
    wav, origin_sr = load_wav(origin_wav_filename)
    if origin_sr != target_sr:
        wav = resample_wav(wav, origin_sr, target_sr)
        if not os.path.exists(target_wav_filename):
            wav_path, _ = os.path.split(target_wav_filename)
            if not os.path.exists(wav_path):
                os.makedirs(wav_path)
            save_wav(wav, target_sr, target_wav_filename)

    # Save linspec
    linspec_filename = target_wav_filename.replace(".wav", "_linspec.pt")
    if not os.path.exists(linspec_filename):
        wav = to_float_tensor(wav)
        wav = devide_by_max_wav_value(wav, max_wav_value)
        linspec = wav_to_linspec(wav, n_fft, hop_length, win_length)
        torch.save(linspec, linspec_filename)


def main():

    OUTER_JSON_DIRS = [
        "./dummies/json/TL4/",
        "./dummies/json/TL22/",
    ]
    ORIGIN_OUTER_WAV_DIRS = [
        "/data/hun/tts/wav_origin/TS4/",
        "/data/hun/tts/wav_origin/TS22/",
    ]
    TARGET_OUTER_WAV_DIRS = [
        "./dummies/wav/TS4/",
        "./dummies/wav/TS22/",
    ]
    TARGET_SR = 22050
    MAX_WAV_VALUE = 32768.0
    N_FFT = 1024
    HOP_LENGTH = 256
    WIN_LENGTH = 1024
    TRAIN_CLEANED_FILENAME = "./dummies/filelist/ver1_train_cleaned.txt/"
    VAL_CLEANED_FILENAME = "./dummies/filelist/ver1_val_cleaned.txt/"
    CLEANED_LOG_FILENAME = "./logs/cleaned_log.log"
    MAKE_NEW_CLEANED_FILE = True
    MAKE_NEW_LOG_FILE = True

    # Resample wav & Save resampled wav & Save linspec
    for origin_outer_wav_dir, target_outer_wav_dir in zip(ORIGIN_OUTER_WAV_DIRS, TARGET_OUTER_WAV_DIRS):
        print(f"======= {origin_outer_wav_dir} =======")
        outer_pbar = tqdm(os.listdir(origin_outer_wav_dir))
        for inner_wav_dir in outer_pbar:
            outer_pbar.set_description(inner_wav_dir)
            inner_wav_dir = os.path.join(origin_outer_wav_dir, inner_wav_dir)
            inner_pbar = tqdm(os.listdir(inner_wav_dir))
            for origin_wav_filename in inner_pbar:
                origin_wav_filename = os.path.join(inner_wav_dir, origin_wav_filename)
                target_wav_filename = origin_wav_filename.replace(origin_outer_wav_dir, target_outer_wav_dir)
                save_resampled_wav_linspec(
                    origin_wav_filename, target_wav_filename, TARGET_SR, MAX_WAV_VALUE, N_FFT, HOP_LENGTH, WIN_LENGTH
                )

    # Make cleaned file
    if MAKE_NEW_CLEANED_FILE:
        if os.path.exists(TRAIN_CLEANED_FILENAME):
            os.remove(TRAIN_CLEANED_FILENAME)
        if os.path.exists(VAL_CLEANED_FILENAME):
            os.remove(VAL_CLEANED_FILENAME)
        print("Remove previous cleaned file.")
    if MAKE_NEW_LOG_FILE:
        if os.path.exists(CLEANED_LOG_FILENAME):
            os.remove(CLEANED_LOG_FILENAME)
        print("Remove previous log file.")

    log_dir_path, log_filename = os.path.split(CLEANED_LOG_FILENAME)
    logger = get_logger(log_dir_path, log_filename)

    print(f"Preprocess {OUTER_JSON_DIRS} ...")
    with open(TRAIN_CLEANED_FILENAME, "a") as train_file:
        for outer_json_dir, outer_wav_dir in zip(OUTER_JSON_DIRS, TARGET_OUTER_WAV_DIRS):
            print(f"======= {outer_json_dir} =======")
            json_dir_paths = os.listdir(outer_json_dir)
            pbar = tqdm(json_dir_paths)
            for json_dir_path in pbar:
                pbar.set_description(f"{json_dir_path}")
                wav_dir_path = os.path.join(outer_wav_dir, json_dir_path)
                json_dir_path = os.path.join(outer_json_dir, json_dir_path)
                if not os.path.exists(wav_dir_path):
                    logger.info(f"wav_dir_path {wav_dir_path} doesn't exist.")
                    continue
                preprocessed = preprocess_one_speaker(json_dir_path, wav_dir_path, log_cleansing=True, logger=logger)
                train_file.write(preprocessed)

    with open(TRAIN_CLEANED_FILENAME, "r") as train_file, open(VAL_CLEANED_FILENAME, "w") as val_file:
        for line in train_file:
            val_file.write(line)
            break


if __name__ == "__main__":
    main()
