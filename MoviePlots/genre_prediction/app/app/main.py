from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.routing import Route
import uvicorn
import os
import logging
import pickle
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Load artifacts
OUTPUT_DIR = 'app/output'
tokenizer = RobertaTokenizer.from_pretrained(OUTPUT_DIR)
model = RobertaForSequenceClassification.from_pretrained(OUTPUT_DIR)
with open(OUTPUT_DIR+'/classes.pkl', 'rb') as fp:
    classes = pickle.load(fp)

# Needed to avoid cross-domain issues
response_header = {
    'Access-Control-Allow-Origin': '*'
}

async def predict(request):
    """Predict the probabilities of genres being in the text."""
    if request.method == 'GET':
        params = request.query_params
    elif request.method == 'POST':
        params = await request.json()
    input_ids = torch.tensor(tokenizer.encode(params['plot'], add_special_tokens=True)).unsqueeze(0)
    logits = model(input_ids)[0]
    probs = torch.sigmoid(logits).detach().cpu().numpy().tolist()[0]
    return JSONResponse({'probs': probs, 'classes': classes}, headers=response_header)

async def homepage(request):
    """Return HTML homepage."""
    return templates.TemplateResponse('index.html', {'request': request})

routes = [
    Route("/", endpoint=homepage),
    Route("/predict", endpoint=predict, methods=["GET", "POST"]),
]

app = Starlette(routes=routes, debug=True)
app.mount('/static', StaticFiles(directory='app/static'))
templates = Jinja2Templates(directory='app/templates')

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), log_level="info")