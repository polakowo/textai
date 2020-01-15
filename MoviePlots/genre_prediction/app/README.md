#### Usage

- Mount Google Drive on your computer
- Create `Colab Notebooks` directory
- Run the following commands in terminal
```
cd "~/Google Drive/Colab Notebooks"
git clone https://github.com/polakowo/textai.git
```
- Run the [DataPreparation notebook](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/MoviePlots/DataPreparation.ipynb) in Google Colab
- Choose a model and run the [Training notebook](https://nbviewer.jupyter.org/github/polakowo/textai/blob/master/MoviePlots/genre_prediction/RoBERTa/Training.ipynb) in Google Colab
- Copy the generated `output` directory to the `app/app` directory (you can delete `models` subdir)
- Run the following commands in terminal
```
docker build -t genre-prediction .
docker run -p 5000:5000 genre-prediction
open http://localhost:5000
```

![Web app screenshot](app.png)
