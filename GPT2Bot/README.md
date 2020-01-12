## GPT2Bot

GPT2Bot implements a decoder ([link](https://github.com/polakowo/textai/blob/master/GPT2Bot/decoder.py)) for [DialoGPT](https://github.com/microsoft/DialoGPT) (based on transformers 2.3.0), an interactive multiturn chatbot ([link](https://github.com/polakowo/textai/blob/master/GPT2Bot/interactive_bot.py)), and a Telegram chatbot ([link](https://github.com/polakowo/textai/blob/master/GPT2Bot/telegram_bot.py)).

### How to use it?

- Install [pytorch](https://github.com/pytorch/pytorch) (tested on 1.2.0)
- Install [transformers](https://github.com/huggingface/transformers) (tested on 2.3.0)
- Install [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) (tested on 12.3.0)

#### Interactive chatbot

- Set your parameters in dialog.cfg
- Run `python interactive_bot.py`

#### Telegram chatbot

- Create a new Telegram bot via BotFather (see https://core.telegram.org/bots)
- Set your parameters such as API token in dialog.cfg
- Run `python telegram_bot.py`

A good thing about Google Colab is free GPU. So why not running the Telegram bot there, for blazingly fast chat? Run the notebook at daytime and do not forget to stop it at night.

[A Colab interactive notebook](https://colab.research.google.com/drive/1ahoqOyoIA7yIfCRm7UaWBeVfm_FADLJt)
