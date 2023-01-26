import os
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, default="/data_raid0/TADEV_BIG_DATA/ASR/TTS/sampled/json")
    parser.add_argument(
        "--destination_path", type=str, default="/data_raid0/TADEV_BIG_DATA/ASR/TTS/sampled/dataset.txt"
    )
    return parser.parse_args()


def make_txt(json_path: str):
    with open(json_path) as f:
        json_data = json.load(f)
        txt = ""
        NumberOfSpeaker = json_data["기본정보"]["NumberOfSpeaker"]

        if json_data["전사정보"]["RefinedLabelText"]:
            text = json_data["전사정보"]["RefinedLabelText"]
        else:
            text = json_data["전사정보"]["TransLabelText"]
        wav_path = json_path.replace("json", "wav")
        txt = wav_path + "|" + NumberOfSpeaker + "|" + text
        return txt


def main(args):
    txt_list = []
    json_dir = args.json_dir
    destination_path = args.destination_path
    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in filenames:
            json_path = os.path.join(dirpath, filename)
            txt_list.append(make_txt(json_path))

    with open(destination_path, "w") as f:
        for txt in txt_list:
            f.write(txt + "\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
