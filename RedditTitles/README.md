#### Generating fake Reddit titles

In this project, I fine-tuned transformers on Reddit titles to produce new, creative ones.

Tags: Reddit, Text Generation, Python, GPT-2, Google Colab, Docker, App

#### Dataset

The dataset consists of 1,488,963 titles of [r/Showerthoughts](https://www.reddit.com/r/Showerthoughts/) subreddit, but you can specify any other subreddit as well.

#### Data preparation

The data preparation process combines data acquisition, data exploration, and data cleansing steps to produce a clean dataset suitable for text generation. After Reddit has disabled Cloudsearch for all of their users, to download the dataset, I used a third-party dataset API called Pushshift (pushshift.io). In particular, I created a method to recursively fetch data from Pushshift using timestamps. After getting the raw data, various preprocessing steps on titles were performed, such as duplicates removal. Titles were also checked for offensive language using [profanity-check](https://pypi.org/project/profanity-check/) and non-English titles using [langdetect](https://pypi.org/project/langdetect/).

[Notebook](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/RedditTitles/GPT2-small/DataPreparation.ipynb)

#### Title generation

Currently, only the smallest version of GPT-2 was fine-tuned for testing purposes. But you could easily change the model size by setting a couple of variables (`MODEL_NAME`, `MODEL_TYPE`) in the trained notebook. A medium-sized model would perform much better!

In the model folder there are two different training notebooks:
- [Training](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/RedditTitles/GPT2-small/Training.ipynb) - performs LM fine-tuning using methods suggested by Hugging Face in [lm_finetuning.py](https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py).
- [Training-fastai](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/RedditTitles/GPT2-small/Training-fastai.ipynb) - performs LM fine-tuning using fastai.
