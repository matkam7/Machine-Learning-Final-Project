# Machine-Learning-Final-Project

```
usage: python3 src/run.py [-h] [-n] [-m] [-r]

optional arguments:
  -h, --help  show this help message and exit
  -n, --nn    Run all neural network models
  -m, --ml    Run machine learning models
  -r, --rnn   Run RNN models
```

## How to run the application:
1. Extract the voice data into a folder named data. In this folder, there should also be a csv file that contains a "filename" column and "sex" column. Here is the link to the dataset that was used for this project:
   
https://www.kaggle.com/rtatman/speech-accent-archive?select=speakers_all.csv

2. Preprocess the audio files for the models that you want to run. If you are planning on running all models, here are the commands that you must run:
```
python3 src/preprocess_data.py --rnn
python3 src/preprocess_data.py --nn
python3 src/preprocess_data.py --ml
```

3. Run the model! If you want to run all models, run the following command:
```
python3 src/run.py
```



