#!/usr/bin/env python
# -*- coding: utf-8 -*-

# !pip install python-telegram-bot --upgrade
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ChatAction, ParseMode
from functools import wraps
from dialog import download_model_folder, load_model, generate_response
import configparser
import logging

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets

def start_command(update, context):
    context.chat_data['turns'] = []
    update.message.reply_text("Hello!")

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

send_typing_action = send_action(ChatAction.TYPING)

@send_typing_action
def message(self, update, context):
    # Parse parameters
    turns_memory = self.config.getint('dialog', 'turns_memory')
    turns = context.chat_data['turns']

    if turns_memory == 0:
        # If you still get different responses then set seed
        context.chat_data['turns'] = []
    # A single turn is a group of user messages and bot responses right after
    turn = {
        'user_messages': [],
        'bot_messages': []
    }
    turns.append(turn)
    turn['user_messages'].append(update.message.text)
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
    update.message.reply_text(bot_message)

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
        self.updater = Updater(config.get('telegram', 'token'), use_context=True)
        dp = self.updater.dispatcher

        # on different commands - answer in Telegram
        # conversation with bot
        dp.add_handler(MessageHandler(Filters.text, self_decorator(self, message)))

        # dialog settings
        dp.add_handler(CommandHandler('start', start_command))

        # log all errors
        dp.add_error_handler(error)

    def run(self):
        logger.info("Running the bot...")

        # Start the Bot
        self.updater.start_polling()

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.updater.idle()

def main():
    config = configparser.ConfigParser(allow_no_value=True)
    config.read('dialog.cfg')

    # Download model artifacts
    target_dir = download_model_folder(config)

    # Load model and tokenizer
    model, tokenizer = load_model(target_dir, config)

    # Run Telegram bot
    bot = TelegramBot(model, tokenizer, config)
    bot.run()
    
if __name__ == '__main__':
    main()
