# tadev-TTS
Text to Speech 솔루션 by TA개발팀 <br />
해당 논문의 내용을 참고하여 작성됨 <br />
 - ["Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"](https://arxiv.org/abs/2106.06103) <br />

## Requirements and Installation

 - PyTorch version == 1.6.0
 - Python version >= 3.7
   - Python 3.6은 2021년 12월 경 보안 지원이 종료되었으며, 22년 7월 기준으로 Microsoft Store 및 VSCode에서 지원이 종료됩니다.
 - To install tadev-TTS and develop locally:
```bash
git clone https://github.com/42maru-ai/tadev-TTS.git

cd tadev-TTS/monotonic_align
python setup.py build_ext

## install essential library
pip install -r requirements.txt
```

## Instructions
 - We support script examples to execute code easily(check `scripts` folder)
 - Following this instruction give you exact matched results.
