<img width=350 src="https://github.com/polakowo/transformers/blob/master/MoviePlots/movie-plots.jpg?raw=true"/>

#### Generating fake movie plots and detecting genres

In this project, transformers are fine-tuned to perform two different tasks: generate new movie plots and predict genres. Movie plots have the advantage of being relatively short, have useful metadata (such as genres), and have a unique writing style that can be easily learned by a machine. The dataset contains short summaries of 117,352 movies and series from around the world.

Tags: *Plots, Genres, Text Generation, Multilabel Classification, Python, GPT-2, DistilGPT2, BERT, RoBERTa, Distillation, Google Colab, Docker, App*

This project is split into three parts: data preparation, text generation and multilabel classification.

#### Data preparation

The goal of this data preparation step is to produce clean data and to engineer features that are ready to be used for training both generators and classifiers. For this, download the data first; then, for each plot in the dataset, check if it's in English, and extract the primary title, the secondary title, the year, and the summary. Save the resulting data in a generic format to the disk and import this data in the respective training notebook.

- [Data preparation](https://nbviewer.jupyter.org/github/polakowo/transformers/blob/master/MoviePlots/DataPreparation.ipynb)

#### Plot generation

Text generation isn't an easy task, and so it requires some experimentation with models and hyperparameters. I tried multiple models, each becoming it's own directory with training and evaluation notebooks. After each fine-tuning process, dumps using various temperatures and top p's were generated. Each training notebook performs tokenization, fine-tuning, and dump generation steps. Each evaluation notebook evaluates the generated dumps; for example, it explores how similar the generated texts are to the training texts semantically using the universal sentence embeddings developed by Google. It explores how uniqueness progresses from primary titles to secondary titles, and summaries. It also explores how temperature and top p affects the generation process.

Plot generation is done in two ways: with titles and without. Including titles is fun but makes the model put too much attention on titles than on genres while training, which leads to overfitting. The analysis of this is done in the respective evaluation notebooks.

- With titles:
  - [Fine-tuning GPT-2 (gpt2simple)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/with-titles/GPT-2-gpt2simple) - It was interesting to observe how the gpt2simple library differs from the huggingface library. For example, gpt2simple uses sampling without replacement, such that some data is selected multiple times while other is never selected. Also, gpt2simple seems to offer faster training and inference. But TensorFlow 2.0 is currently not supported.
  - [Fine-tuning GPT-2 (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/with-titles/GPT-2)
  - [Web application for generating plots and titles (huggingface only)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/with-titles/app)
- Without titles:
  - [Fine-tuning GPT-2 (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/without-titles/GPT-2)
  - [Fine-tuning Distilled GPT-2 (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/without-titles/GPT-2) - Distilled GPT-2 ([link](https://github.com/huggingface/transformers/tree/master/examples/distillation)) has less parameters than the original GPT-2 architecture. This leads to generated texts being simple and sometimes repeated in a loop. 
  - [Web application for generating plots (huggingface only)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/without-titles/app)

#### Genre prediction

Because each plot can be assigned to multiple genres, genre prediction is a multilabel classification task. Some genres are more prevalent than the others, which leads to a class imbalance problem. It was tackled by passing weights to the BCE loss rather than resampling. 

- [Fine-tuning RoBERTa (fastai + huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/genre_prediction/RoBERTa) - Using fastai for this task is much more easier than tweaking the script [run_glue.py](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py).
- [Fine-tuning LM and classification BERT (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/genre_prediction/BERT/lm_finetuning) - Language model fine-tuning was done with with a maximum block size of 256 and linear schedule. A block size of 512 produced no improvement in loss; I still can't figure out why. 
- [Web application for detecting genres (huggingface only)](https://github.com/polakowo/transformers/tree/master/MoviePlots/genre_prediction/app)

#### Usage

Every notebook is meant to be run in Google Colab. For this, mount Google Drive on your computer, create a folder "Colab Notebooks", and pull the transformers repository there. Then run the [DataPreparation.ipynb](https://nbviewer.jupyter.org/github/polakowo/transformers/blob/master/MoviePlots/DataPreparation.ipynb) notebook to prepare the data. After this, you should be ready to execute any Training notebook with GPU support. Tip: Do not abuse Colab, remember to terminate any unused GPU sessions, and Colab will award you with fastest devices.
