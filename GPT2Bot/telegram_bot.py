#  Copyright (c) polakowo
#  Licensed under the MIT license.

# !pip install python-telegram-bot --upgrade
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ChatAction, ParseMode
from functools import wraps
import configparser
import argparse
import logging
import requests
from urllib.parse import urlencode
from requests.exceptions import HTTPError

from model import download_model_folder, load_model
from decoder import generate_response

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets

def restart_command(update, context):
    context.chat_data['turns'] = []
    update.message.reply_text("Let's start from scratch.")

def help_command(update, context):
    update.message.reply_text("Just start texting me. If it's getting annoying, type /restart")

def translate_message_to_gif(message, config):
    # https://engineering.giphy.com/contextually-aware-search-giphy-gets-work-specific/
    params = {
        'api_key': config.get('chatbot', 'giphy_token'),
        's': message,
        'weirdness': config.getint('chatbot', 'giphy_weirdness')
    }
    url = "http://api.giphy.com/v1/gifs/translate?" + urlencode(params)
    try:
        response = requests.get(url)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')
    return response.json()['data']['images']['fixed_height']['url']

def self_decorator(self, func):
    """Passes bot object to func command."""
    # TODO: Any other ways to pass variables to handlers?
    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)
    return command_func

def send_action(action):
    """Sends `action` while processing func command."""
    def decorator(func):
        @wraps(func)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(self, update, context, *args, **kwargs)
        return command_func
    return decorator

def gif_mode(update, context):
    context.chat_data['gif_mode'] = True
    update.message.reply_text("Switched to the GIF mode.")

def text_mode(update, context):
    context.chat_data['gif_mode'] = False
    update.message.reply_text("Switched to the text mode.")

send_typing_action = send_action(ChatAction.TYPING)

@send_typing_action
def message(self, update, context):
    # Parse parameters
    turns_memory = self.config.getint('chatbot', 'turns_memory')
    if 'turns' not in context.chat_data:
        context.chat_data['turns'] = []
    turns = context.chat_data['turns']

    if context.chat_data.get('gif_mode', False):
        # TODO: Find a way to extract caption from GIFs sent by the user
        user_message = update.message.text
    else:
        user_message = update.message.text
    if turns_memory == 0:
        # If you still get different responses then set seed
        context.chat_data['turns'] = []
    # A single turn is a group of user messages and bot responses right after
    turn = {
        'user_messages': [],
        'bot_messages': []
    }
    turns.append(turn)
    turn['user_messages'].append(user_message)
    # Merge turns into a single history (don't forget EOS token)
    history = ""
    from_index = max(len(turns)-turns_memory-1, 0) if turns_memory >= 0 else 0
    for turn in turns[from_index:]:
        # Each turn begings with user messages
        for message in turn['user_messages']:
            history += message + self.tokenizer.eos_token
        for message in turn['bot_messages']:
            history += message + self.tokenizer.eos_token

    # Generate bot messages
    bot_message = generate_response(self.model, self.tokenizer, history, self.config)
    turn['bot_messages'].append(bot_message)
    # Return response as text
    update.message.reply_text(bot_message)
    if context.chat_data.get('gif_mode', False):
        # Return response as GIF
        gif_url = translate_message_to_gif(bot_message, self.config)
        context.bot.send_animation(update.effective_message.chat_id, gif_url)

def error(update, context):
    logger.warning(context.error)

class TelegramBot:
    def __init__(self, model, tokenizer, config):
        logger.info("Initializing the bot...")

        # Set global variables
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Set up Telegram bot
        self.updater = Updater(config.get('chatbot', 'telegram_token'), use_context=True)
        dp = self.updater.dispatcher

        # on different commands - answer in Telegram
        # conversation with bot
        dp.add_handler(MessageHandler(Filters.text, self_decorator(self, message)))

        # chatbot settings
        dp.add_handler(CommandHandler('restart', restart_command))
        dp.add_handler(CommandHandler('help', help_command))
        dp.add_handler(CommandHandler('gifmode', gif_mode))
        dp.add_handler(CommandHandler('textmode', text_mode))

        # log all errors
        dp.add_error_handler(error)

    def run_chat(self):
        logger.info("Running the chatbot...")

        # Start the Bot
        self.updater.start_polling()

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.updater.idle()

def main():
    # Script arguments can include path of the config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default="chatbot.cfg")
    args = arg_parser.parse_args()

    # Read the config
    config = configparser.ConfigParser(allow_no_value=True)
    with open(args.config) as f:
        config.read_file(f)

    # Download model artifacts
    target_dir = download_model_folder(config)

    # Load model and tokenizer
    model, tokenizer = load_model(target_dir, config)

    # Run Telegram bot
    bot = TelegramBot(model, tokenizer, config)
    bot.run_chat()
    
if __name__ == '__main__':
    main()
