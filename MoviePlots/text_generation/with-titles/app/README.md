#### App deployment

- Mount Google Drive on your computer
- Create `Colab Notebooks` directory

In terminal:
- `git clone https://github.com/polakowo/transformers.git`

In Google Colab:
- Run the [DataPrep notebook](https://nbviewer.jupyter.org/github/polakowo/transformers/blob/master/MoviePlots/DataPrep.ipynb)
- Choose a model and run the [Training notebook](https://nbviewer.jupyter.org/github/polakowo/transformers/blob/master/MoviePlots/text_generation/with-titles/GPT-2/Training.ipynb)
- Copy `output` directory to the `app/app` directory

In terminal:
```
docker build -t text-generation .
docker run -p 5000:5000 text-generation
open http://localhost:5000
```
