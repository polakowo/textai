#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.

import os
import requests
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
import configparser
import logging

# !pip install transformers==2.3.0
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# If you get tensorflow deprecation warnings, run
# pip uninstall numpy
# pip install numpy==1.16.4

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Note that the model size is roughly half of the GPT model because our model is saved by fp16
LSP_MODEL_URL = {
    'multiref': {
        'large_fs': 'https://convaisharables.blob.core.windows.net/lsp/multiref/large_fs.pkl',
        'medium_fs': 'https://convaisharables.blob.core.windows.net/lsp/multiref/medium_fs.pkl',
        'medium_ft': 'https://convaisharables.blob.core.windows.net/lsp/multiref/medium_ft.pkl',
        'small_fs': 'https://convaisharables.blob.core.windows.net/lsp/multiref/small_fs.pkl',
        'small_ft': 'https://convaisharables.blob.core.windows.net/lsp/multiref/small_ft.pkl'
    },
    'dstc': {
        'medium_ft': 'https://convaisharables.blob.core.windows.net/lsp/DSTC/medium_ft.pkl'
    }
}

CONFIG_FILE = {
    'small': 'https://convaisharables.blob.core.windows.net/lsp/117M/config.json',
    'medium': 'https://convaisharables.blob.core.windows.net/lsp/345M/config.json',
    'large': 'https://convaisharables.blob.core.windows.net/lsp/1542M/config.json'
}

VOCAB_FILE = {
    'small': 'https://convaisharables.blob.core.windows.net/lsp/117M/vocab.json',
    'medium': 'https://convaisharables.blob.core.windows.net/lsp/345M/vocab.json',
    'large': 'https://convaisharables.blob.core.windows.net/lsp/1542M/vocab.json'
}

MERGE_FILE = {
    'small': 'https://convaisharables.blob.core.windows.net/lsp/117M/merges.txt',
    'medium': 'https://convaisharables.blob.core.windows.net/lsp/345M/merges.txt',
    'large': 'https://convaisharables.blob.core.windows.net/lsp/1542M/merges.txt'
}

def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def download_file(url, folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    file_name = os.path.basename(url)
    if 'pytorch_model.bin' in file_name:
        file_name = 'pytorch_model.bin'

    if os.path.isfile(os.path.join(folder, file_name)):
        return

    with open(os.path.join(folder, file_name), 'wb') as f:
        http_get(url, f)


def download_model_folder(config):
    # Parse parameters
    data_folder = config.get('model', 'data_folder')
    model_size = config.get('model', 'model_size')
    dataset = config.get('model', 'dataset')
    from_scratch = config.getboolean('model', 'from_scratch')

    logger.info("Downloading model files...")
    # Create data folder if needed
    if not os.path.exists(data_folder):
        os.makedirs(data_folder, exist_ok=True)
    # Build target folder name (must be unique across all parameter combinations)
    target_folder_name = model_size + "_" + dataset + ("_fs" if from_scratch else "_ft")
    target_folder = os.path.join(data_folder, target_folder_name)
    # Download files
    download_file(CONFIG_FILE[model_size], target_folder)
    download_file(VOCAB_FILE[model_size], target_folder)
    download_file(MERGE_FILE[model_size], target_folder)
    model_train_type = model_size + ('_fs' if from_scratch else '_ft')
    if model_train_type not in LSP_MODEL_URL[dataset]:
        k = ','.join(list(LSP_MODEL_URL[dataset].keys()))
        raise ValueError(f"'{model_train_type}' not exist for dataset '{dataset}', please choose from [{k}]")
    download_file(LSP_MODEL_URL[dataset][model_train_type], target_folder)
    return target_folder

def load_model(target_folder, config):
    # Parse parameters
    model_size = config.get('model', 'model_size')

    logger.info("Loading the model...")
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    # Tokenizer
    tokenizer = GPT2Tokenizer(os.path.join(target_folder, 'vocab.json'), os.path.join(target_folder, 'merges.txt'))
    # Config
    config = GPT2Config.from_json_file(os.path.join(target_folder, 'config.json'))
    # Weights
    state_dict_path = glob(os.path.join(target_folder, f'*.pkl'))[0]
    state_dict = torch.load(state_dict_path, map_location=device)
    if model_size == 'small':
        for key in list(state_dict.keys()):
            state_dict[key.replace('module.', '')] = state_dict.pop(key)
    state_dict['lm_head.weight'] = state_dict['lm_head.decoder.weight']
    state_dict.pop("lm_head.decoder.weight", None)
    # Model
    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, tokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, context, config):
    # Parse parameters
    num_samples = config.getint('dialog', 'num_samples')
    length = config.getint('dialog', 'length')
    temperature = config.getfloat('dialog', 'temperature')
    top_k = config.getint('dialog', 'top_k')
    top_p = config.getfloat('dialog', 'top_p')
    no_cuda = config.getboolean('dialog', 'no_cuda')

    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0.0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

def generate_response(model, tokenizer, context, config):
    # Parse parameters
    num_samples = config.getint('dialog', 'num_samples')
    seed = config.get('dialog', 'seed')
    seed = int(seed) if seed is not None else None

    # Make answers reproducible only if wanted
    if seed is not None:
        set_seed(seed)

    # Generate response
    context_tokens = tokenizer.encode(context)
    out = sample_sequence(model, context_tokens, config)
    out = out[:, len(context_tokens):].tolist()
    texts = []
    for o in out:
        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        text = text[: text.find(tokenizer.eos_token)]
        texts.append(text)
    if num_samples == 1:
        return texts[0]
    return texts

def run_dialog(model, tokenizer, config):
    # Parse parameters
    turns_memory = config.getint('dialog', 'turns_memory')

    logger.info("Running the dialog...")
    turns = []
    while True:
        prompt = input("User >>> ")
        if turns_memory == 0:
            # If you still get different responses then set seed
            turns = []
        if prompt == '/start_over':
            turns = []
            continue
        if prompt == '/quit':
            break
        # A single turn is a group of user messages and bot responses right after
        turn = {
            'user_messages': [],
            'bot_messages': []
        }
        turns.append(turn)
        turn['user_messages'].append(prompt)
        # Merge turns into a single history (don't forget EOS token)
        history = ""
        from_index = max(len(turns)-turns_memory-1, 0) if turns_memory >= 0 else 0
        for turn in turns[from_index:]:
            # Each turn begings with user messages
            for message in turn['user_messages']:
                history += message + tokenizer.eos_token
            for message in turn['bot_messages']:
                history += message + tokenizer.eos_token

        # Generate bot messages
        bot_message = generate_response(model, tokenizer, history, config)
        print("Bot >>>", bot_message)
        turn['bot_messages'].append(bot_message)

def main():
    config = configparser.ConfigParser(allow_no_value=True)
    config.read('dialog.cfg')

    # Download model artifacts
    target_dir = download_model_folder(config)

    # Load model and tokenizer
    model, tokenizer = load_model(target_dir, config)

    # Run dialog with GPT-2
    run_dialog(model, tokenizer, config)
    

if __name__ == '__main__':
    main()
