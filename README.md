# Metric Learning Pipelines for Speaker-Diarization
Speaker diarization using Deep Attention Model embeddings and metric learning. The variable parts of the architecture namely the negative sampling techniques (Random, Semi-Hard, Distance Weighted Sampling), types of loss (Triplet and Quadruplet) and margins (Fixed and Adaptive) are investigated to check for performance improvements in the given pipeline.

See the diagram below for a summary of the approach.

![Metric Learning Pipeline](https://github.com/vivsivaraman/Metric-Learning-Pipelines-for-Speaker-Diarization/blob/master/approach.png)

## Requirements
* python 3.6
* numpy >= 1.11.0
* tensorflow >= 1.5.0
* scikit-learn >= 0.18
* matplotlib >= 2.1.0
* pyannote.core >= 1.3.1
* pyannote.metrics >= 1.6.1
* python-speech-features >= 0.6
* sox >= 1.3.2
* h5py >= 2.6.0

## Prerequisites
The TEDLIUM Corpus used for training is available at http://www.openslr.org/7/
The CALLHome conversational speech corpus for testing is available at https://media.talkbank.org/ca/CallHome/. Use wget command to download the data for the different languages.

The required libraries can be installed using pip install -r requirements.txt

The corpora directory needs to be in the folder
```
/Data
```

Download the vbs_demo package from https://www.voicebiometry.org/download/vbs_demo.tgz and put in current directory.

## Data preprocessing
### TEDLIUM Corpus
Specify processing steps in ```process_ted.py``` and execute the same to obtain the MFCC segments for every recording.

Optionally build the train subset and development set with ```generate_ted_subset.py``` by specifying the respective paths.
### CALLHOME Corpus
Specify processing steps in ```process_callhome.py``` and execute the same to obtain the MFCC segments.

### Model training parameters
To specify the paths, network and training configurations,sampling type, type of loss, margin and other parameters,modify:
```
hyperparams.py
```
## Training

Run ```run_metriclearn.py``` which saves the models evaluated at different steps in the log dir. The code also provides the list of training losses at every global step

## Testing and Obtaining Diarization Metrics
First extract the embeddings from the trained model with
```run_testembeddings.py``` evaluated at the checkpoint of the least validation loss (dev_history.csv).

To perform diarization clustering:
Execute ```run_diarization.py``` which stores the Diarization Error Rates in a .csv file.


### Run the demo

Download the pre-extracted embeddings for the English language corpus from the CALLHOME dataset located in Data.tar CALLHOME folder.  It can be used to execute ```run_diarization.py``` to obtain the DERs.


## Citations

Our papers are cited as:

```
@INPROCEEDINGS{narayanaswamyspd,
  author={V. S. {Narayanaswamy} and J. J. {Thiagarajan} and H. {Song} and A. {Spanias}},
  booktitle={ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={Designing an Effective Metric Learning Pipeline for Speaker Diarization},
  year={2019},
  volume={},
  number={},
  pages={5806-5810}
}

@inproceedings{Song2018,
  author={Huan Song and Megan Willi and Jayaraman J. Thiagarajan and Visar Berisha and Andreas Spanias},
  title={Triplet Network with Attention for Speaker Diarization},
  year=2018,
  booktitle={Proc. Interspeech 2018},
  pages={3608--3612},
  doi={10.21437/Interspeech.2018-2305},
  url={http://dx.doi.org/10.21437/Interspeech.2018-2305}
}
```
