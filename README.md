# revcaptcha-service

A voice authentication system that uses the reverse voice CAPTCHA mechanism to authenticate users. It uses a two-stage sequential design which uses a lexical-matching stage and identity-matching stage to verify an authentication attempt. A user must enroll to the system by providing 5 or 10 voice recordings which are used as an enrollment centroid\vocal profile of that user. In an authentication attempt, the user first have to read out each letter of a randomized 5-character alphanumeric prompt that the system provides to them. The user must read out this prompt in 10 seconds. The lexical gate checks if the content of the user's audio and the randomized text prompt matches exactly. If so, the response audio will move on to the next stage, speaker verification. In speaker verification, a pre-trained speaker embedding model will turn the response audio into a fixed-length embedding. This embedding is scored against the target user's enrollment centroid embedding using cosine similarity. If the score is equal to higher than the threshold, that attempt is authenticated.

## Setup
### Environment setup
Before you do anything, please change the current directory to the cloned folder:

```
cd .\revcaptcha-service\
```

Firstly, I recommend setting up an virtual environment in order to not clash with your global or other packages since the project needs different versions of packages to work. 
You can use  a venv virtual environment by typing or pasting this command at the terminal:

```
py -3.11 -m venv .venv
```

You can set any env name in place of ".venv"

If you prefer to use a Conda environment (from Anaconda 3), please use this command to setup:

```
conda create -n myenv python=3.11
```

 You can set any env name in place of "myenv"


Please be noted that you must install and use python 3.11 in any environment in order for the project to work.

You can activate the created virtual environment with venv using the command in the directory that contains the virtual environment folder:

For venv:
On Windows terminal
```
.your-venv-name\Scripts\activate
``` 

On Windows Powershell
```
.your-venv-name\Scripts\activate.ps1
``` 

On macOS or Unix:
```
source .your-venv-name/bin/activate
``` 

For conda:
```
conda activate your-conda-name
``` 


### Dependencies
After activting the virtual environment, you must install the necessary requirements (including packages and dependencies) using the following command:

```
pip install .
```

If you also want the development tools:

```
pip install -e ".[dev]"
```
**IMPORTANT:** Please check if both torch and torch audio are at versions 2.8.0 by using 
```
pip show torch torchaudio
```
If they are not, please reinstall using:
```
pip install torch=2.8.0 torchaudio=2.8.0
```


## What works so far
### Code Generator
- Generates a randomized 5-character alphanumeric code using a securer randomizer seed
- Generates the code with a fair chance for all possible combinations

```
cd .\revcaptcha-service\src\revcaptcha\
python challenge.py
```

### Character Normalizer
- Normalizes the transcribed text from the user using the NORMALIZATION_MAP
- Strips the transcribed text of any whitesapces
- Removes common punctuation from the transcribed text

```
cd .\revcaptcha-service\src\revcaptcha\
python text.py
```

### Lexical Matcher
- Checks for an exact lexical match between the normalized transcribed text and the prompt text
```
cd .\revcaptcha-service\src\revcaptcha\
python text.py
```

### Audio Loader
- Converts a valid audio file between 0.5s and 15.0s duration to a mono-channel 16khz ndarray
- Error handling for audio duration not between 0.5s and 15.0s, unreadable files, files not existing
- Clips the maximum and minimum absolute values to 1 and -1 respectively using np.clip
- Converts any multi-channel audio to mono-channel using np.mean
```
cd .\revcaptcha-service\src\revcaptcha\
python audio.py
```

### Transcriber 
- Converts a valid audio file between 0.5s and 15.0s duration to a string that contains the spoken content
- Uses the Whisper Small model from openai-whisper 
```
cd .\revcaptcha-service\src\revcaptcha\
python asr.py
```

### Embedder
- Converts an valid Numpy array or audio file to a (512,) shaped Numpy array
    - First converts to a tensor using unsqueeze 
    - Scales back to actual values by multiplying each sample with 32768
    - Uses CAM++ model's extract_features_from_pcm method
    - Converts back to an Numpy Array 
```
cd .\revcaptcha-service\src\revcaptcha\
python speaker.py
```

### Scoring
- Builds a voice profile using 5 or 10 enrollment clips of a user
- Calculates the cosine similarity score between two embeddings 
- Checks if the cosine score passes the threshold
```
cd .\revcaptcha-service\src\revcaptcha\
python scoring.py
```


## Models
Currently, we are using openai-whisper's Whisper Small model for our transcribing tasks and and English variant of CAM++ from the Hugging Face repository: ``` Wespeaker/wespeaker-voxceleb-campplus ```. The CAM++ models lives inside ``` models/campplus/ ```

## Lexical Gate passrates on Track H
### Before Map Improvement
#### Dev
dev
Pass Rate: 73.5%
239 out of 325 passes

#### Check
check
Pass Rate: 72.8%
118 out of 162 passes

### After Map Improvement
#### Dev
dev
Pass Rate: 75.1%
244 out of 325 passes

#### Check
check
Pass Rate: 75.9%
123 out of 162 passes

## Peformance on My Machine
**CPU**: AMD Ryzen 7 5700X 8-Core Processor 

| Model | Input | Memory before (MB) | Memory after load (MB) | Load (s) | First call (s) | Average of next 10 (s) |
|---|---|---:|---:|---:|---:|---:|
| Whisper Small | 1 s | 36 | 1599 | 3.71 | 1.265 | 1.067 |
| Whisper Small | 4 s | 36 | 1599 | 3.71 | 1.135 | 1.135 |
| CAM++ | 1 s | 36 | 553 | 9.30 | 0.046 | 0.041 |
| CAM++ | 4 s | 36 | 553 | 9.30 | 0.095 | 0.090 |

## Test Clips
Inside tests\data, we have included clips that are under the license [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/deed.en). You can check the README.md files inside the folder for more details. 

## Running the tests
This is only for installations with development tools. Once you have completed the installation and setup, you can navigate into the tests folder by the following command:

For Windows:
```
cd src\revcaptcha\tests\
pytest 
```
All 211 tests should pass

You can also test the Whisper and CAM++ models using the slow mode (which are deselected by default pytest calls) using:
```
pytest -m slow
```
All 7 tests should pass 

**IMPORTANT:** Please be noted that sometimes a test or two may fail due to the transcription being different in each run. In that case, please rerun the tests again. 
