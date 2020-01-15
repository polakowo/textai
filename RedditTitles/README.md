### RedditTitles

In this project, GPT-2 was fine-tuned on Reddit titles to produce new, creative ones.

Tags: Reddit, Text Generation, Python, GPT-2, Google Colab, Docker, App

#### Dataset

The dataset consists of 1,488,963 titles of [r/Showerthoughts](https://www.reddit.com/r/Showerthoughts/) subreddit, but you can specify any other subreddit as well.

#### Data preparation

The [Data Preparation notebook](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/RedditTitles/GPT2-small/DataPreparation.ipynb) combines data acquisition, data exploration, and data cleansing steps to produce a clean dataset suitable for text generation. After Reddit has disabled Cloudsearch for all of their users, to download the dataset, I used a third-party dataset API called Pushshift (pushshift.io). In particular, I created a method to recursively fetch data from Pushshift using timestamps. After getting the raw data, various preprocessing steps on titles were performed, such as duplicates removal. Titles were also checked for offensive language using [profanity-check](https://pypi.org/project/profanity-check/) and non-English titles using [langdetect](https://pypi.org/project/langdetect/).

#### Title generation

Currently, only the smallest version of GPT-2 was fine-tuned for testing purposes. But you could easily change the model size by setting a couple of variables (`MODEL_NAME`, `MODEL_TYPE`) in the trained notebook. A medium-sized model would perform much better!

In the model folder there are two different training notebooks:
- [Training](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/RedditTitles/GPT2-small/Training.ipynb) - performs LM fine-tuning using methods suggested by Hugging Face in [lm_finetuning.py](https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py).
- [Training-fastai](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/RedditTitles/GPT2-small/Training-fastai.ipynb) - performs LM fine-tuning using fastai.

#### Evaluation

The [Evaluation notebook](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/RedditTitles/GPT2-small/Evaluation.ipynb) evaluates the generated dumps; for example, it explores how similar the generated texts are to the training texts syntactically using Jaccard index on common words and semantically using the [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) developed by Google. The notebook also explores how `temperature` and `top_p` affect the generation process.

#### Usage

Every notebook is meant to be run in Google Colab. For this, mount Google Drive on your computer, create a folder "Colab Notebooks", and pull the transformers repository there. Then run the `DataPreparation.ipynb` notebook to prepare the data. After this, you should be ready to execute any notebook with GPU support. Tip: Do not abuse Colab, remember to terminate any unused GPU sessions, and Colab will award you with fastest devices.
