from jamo import hangul_to_jamo
from typing import List


PAD = ['_']
SPACE = [' ']
PUNC = ['!', ',', '.', '?', '~']
JAMO_LEADS = [chr(_) for _ in range(0x1100, 0x1113)]
JAMO_VOWELS = [chr(_) for _ in range(0x1161, 0x1176)]
JAMO_TAILS = [chr(_) for _ in range(0x11A8, 0x11C3)]
SYMBOLS = PAD + SPACE + PUNC + JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS  # First symbol should be pad

_symbol_to_id = {s: i for i, s in enumerate(SYMBOLS)}
# _id_to_symbol = {i: s for i, s in enumerate(SYMBOLS)}


def text_to_sequence(text: str) -> List[int]:
    """str symbol을 integer로 mapping"""
    global _symbol_to_id
    text = text.replace(u'\xa0', u' ')  # https://stackoverflow.com/questions/10993612/how-to-remove-xa0-from-string-in-python
    try:
        sequence = [_symbol_to_id[symbol] for symbol in hangul_to_jamo(text)]
    except Exception as e:
        print(f"{e} is not in SYMBOLS. return []. (text: {text})")  # # 에러 메시지 수정이 필요할 듯
        sequence = []
    return sequence


# def sequence_to_text(sequence):
#     global _id_to_symbol
#     text = ''.join([_id_to_symbol[id] for id in sequence])
#     return text


def add_blank(token: List[int]) -> List[int]:
    """token사이마다 공백 추가"""
    result = [0] * (len(token) * 2 + 1)
    result[1::2] = token
    return result
