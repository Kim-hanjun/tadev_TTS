from jamo import hangul_to_jamo
from typing import List
import unicodedata
import re
from unidecode import unidecode
from phonemizer import phonemize

# sudo apt install espeak

PAD = ["_"]
SPACE = [" "]
PUNC = ["!", ",", ".", "?", "~", "'", "-"]  # "'", "-" are for english
JAMO_LEADS = [chr(_) for _ in range(0x1100, 0x1113)]
JAMO_VOWELS = [chr(_) for _ in range(0x1161, 0x1176)]
JAMO_TAILS = [chr(_) for _ in range(0x11A8, 0x11C3)]
UPPER_ALPHABETS = [chr(_) for _ in range(65, 91)]
LOWER_ALPHABETS = [chr(_) for _ in range(97, 123)]
SYMBOLS = (
    PAD + SPACE + PUNC + JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + UPPER_ALPHABETS + LOWER_ALPHABETS
)  # First symbol should be pad

_symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}


def cleaning_text(text: str) -> str:
    m = re.findall(r"[\t=+#/\:^$@*\"※~&%ㆍ』\\‘|\(\)\[\]\<\>`\…》]", text)
    for i in m:
        text = text.replace(i, "")
    text = text.strip()
    text = re.sub(" +", " ", text)
    m = set(re.findall(r" [.,!?]", text))
    for i in m:
        text = text.replace(i, i[1])
    if text[-1] not in [".", "?", "!", "~"]:
        text = text + "."
    return text


def text_to_sequence(text: str) -> List[int]:
    """str symbol을 integer로 mapping"""
    global _symbol_to_id
    text = text.replace(
        "\xa0", " "
    )  # https://stackoverflow.com/questions/10993612/how-to-remove-xa0-from-string-in-python
    try:
        sequence = [_symbol_to_id[symbol] for symbol in hangul_to_jamo(text)]
    except Exception as e:
        print(f"{e} is not in SYMBOLS. return []. (text: {text})")  # # 에러 메시지 수정이 필요할 듯
        sequence = []
    return sequence


def add_blank_to_text(token: List[int]) -> List[int]:
    """token사이마다 공백 추가"""
    result = [0] * (len(token) * 2 + 1)
    result[1::2] = token
    return result


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language="en-us", backend="espeak", strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(
        text, language="en-us", backend="espeak", strip=True, preserve_punctuation=True, with_stress=True
    )
    phonemes = collapse_whitespace(phonemes)
    return phonemes
