#### Generating fake movie plots and detecting genres

In this project, we use transformers to perform two different tasks: generate new movie plots and predict genres. Movie plots have the advantage of being relatively short, have useful metadata (such as genres), and have a unique writing style that can be easily learned by a machine. The dataset contains short summaries of 117,352 movies and series from around the world.

Tags: *Plots, Genres, Text Generation, Multilabel Classification, Python, GPT-2, DistilGPT2, BERT, RoBERTa, Distillation, Google Colab, Docker, App*

This project is split into three parts: data preparation, text generation and multilabel classification.

#### Data preparation

- [Data preparation notebook](https://nbviewer.jupyter.org/github/polakowo/transformers/blob/master/MoviePlots/DataPrep.ipynb) - For each plot in the dataset, extract the primary title, the secondary title, the year, and the summary.

#### Plot generation

Plot generation is done in two ways: with titles and without. Including titles is fun but makes the model put too much attention on titles than on genres while training, which leads to overfitting. The analysis of this is done in the respective evaluation notebooks.

- With titles:
  - [Fine-tuning GPT-2 (gpt2simple)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/with-titles/GPT-2-gpt2simple)
  - [Fine-tuning GPT-2 (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/with-titles/GPT-2)
  - [Web application for generating plots (supports huggingface only)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/with-titles/app)
- Without titles:
  - [Fine-tuning GPT-2 (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/without-titles/GPT-2)
  - [Fine-tuning Distilled GPT-2 (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/without-titles/GPT-2)
  - [Web application for generating plots (supports huggingface only)](https://github.com/polakowo/transformers/tree/master/MoviePlots/text_generation/without-titles/app)

#### Genre prediction

- [Fine-tuning RoBERTa (fastai + huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/genre_prediction/RoBERTa)
- [Fine-tuning LM of BERT (huggingface)](https://github.com/polakowo/transformers/tree/master/MoviePlots/genre_prediction/BERT/lm_finetuning)
- [Web application for detecting genres (supports huggingface only)](https://github.com/polakowo/transformers/tree/master/MoviePlots/genre_prediction/app)

<img width=350 src="https://github.com/polakowo/transformers/blob/master/MoviePlots/movie-plots.jpg?raw=true"/>
