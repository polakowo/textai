from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.routing import Route
import uvicorn
import os
import sys
import logging
import gc
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}

# Load artifacts
model = GPT2LMHeadModel.from_pretrained('app/output')
tokenizer = GPT2Tokenizer.from_pretrained('app/output')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

def sample_sequence(prompt="", length=128, num_samples=1, temperature=1, top_k=0, top_p=0.0, 
    repetition_penalty=1.0, stop_token=tokenizer.eos_token):
    """Generate text using pretrained model."""
    set_seed(random.randint(0, 1000)) # makes generated texts non-repetitive

    context_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    context = torch.tensor(context_tokens, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in tqdm(range(length)):

            inputs = {'input_ids': generated}

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    generated = generated[:, len(context_tokens):].tolist()
    texts = []
    for o in generated:
        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        text = text[: text.find(stop_token) if stop_token else None]
        texts.append(text)
    return texts

def generate_text(params):
    """Generate and parse text."""
    
    # Configure generation process
    kwargs = {}
    kwargs['prompt'] = tokenizer.eos_token + str(params['prompt'])
    kwargs['temperature'] = float(params['temperature'])
    kwargs['top_p'] = float(params['top_p'])

    # Generate text
    text = kwargs['prompt'] + sample_sequence(**kwargs)[0]

    # Parse text
    if text.find(tokenizer.eos_token) == 0:
        text = text.split(tokenizer.eos_token)[1]
    if tokenizer.eos_token in text:
        text = text.split(tokenizer.eos_token)[0]

    return {'text': text}

async def generate(request):
    """Generate text and return the parsed result as a dict."""
    if request.method == 'GET':
        params = request.query_params
    elif request.method == 'POST':
        params = await request.json()
    elif request.method == 'HEAD':
        return JSONResponse({'text': ''}, headers=response_header)
    logging.info(params)
    gc.collect()
    return JSONResponse(generate_text(params), headers=response_header)

async def homepage(request):
    """Return HTML homepage."""
    return templates.TemplateResponse('index.html', {'request': request})

routes = [
    Route("/", endpoint=homepage),
    Route("/generate", endpoint=generate, methods=["GET", "POST"]),
]

app = Starlette(routes=routes, debug=True)
app.mount('/static', StaticFiles(directory='app/static'))
templates = Jinja2Templates(directory='app/templates')

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), log_level="info")