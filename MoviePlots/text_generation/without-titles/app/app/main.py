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
from random import uniform
import run_generation

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}

BOS_TOKEN = '<|startoftext|>'
EOG_TOKEN = '<|endofgenre|>'
EOS_TOKEN = '<|endoftext|>'

def generate_text(params):
    """Generate text using transformers."""
    if params['genre'] and params['prefix']:
        prompt = BOS_TOKEN + params['genre'] + EOG_TOKEN + params['prefix']
    elif params['genre']:
        prompt = BOS_TOKEN + params['genre'] + EOG_TOKEN
    elif params['prefix']:
        # If user specifies prefix, the model cannot generate the genre anymore
        prompt = params['prefix']
    else:
        prompt = BOS_TOKEN
    text = run_generation.main([
        '--model_type=gpt2',
        '--model_name_or_path=app/output',
        f"--prompt={prompt}" if prompt else '--prompt=""',
        f'--temperature={float(params["temp"]) if params["temp"] else uniform(0.7, 1)}',
        f'--top_p={float(params["top_p"]) if params["top_p"] else 0}',
        '--num_samples=1',
        '--length=256',
        f'--stop_token={EOS_TOKEN}'
    ])
    return prompt+text

def parse_text(text):
    """Parse text."""
    if BOS_TOKEN in text:
        if text.index(BOS_TOKEN) == 0:
            text = text.split(BOS_TOKEN)[1]
    if EOS_TOKEN in text:
        text = text.split(EOS_TOKEN)[0]
    if EOG_TOKEN in text:
        genre = text.split(EOG_TOKEN)[0]
        text = text.split(EOG_TOKEN)[1]
    else:
        genre = ''
    plot = '.'.join(text.split('.')[:-1])+'.'
    # Remove the last uncomplete sentence
    return {
        'genre': genre.strip(),
        'plot': plot.strip()
    }

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
    return JSONResponse(parse_text(generate_text(params)), headers=response_header)

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