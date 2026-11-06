# revcaptcha-service

A voice authentication system that uses the reverse voice CAPTCHA mechanism to authenticate users. It uses a two-stage sequential design which uses a lexical-matching stage and identity-matching stage to verify an authentication attempt. A user must enroll to the system by providing 3 to 10 voice recordings which are used as an enrollment centroid\vocal profile of that user. In an authentication attempt, the user first have to read out each letter of a randomized text prompt that the system provides to them. The lexical gate checks if the content of the user's audio and the randomized text prompt matches exactly. If so, the response audio will move on to the next stage, speaker verification. In speaker verification, a pre-trained speaker embedding model will turn the response audio into a fixed-length embedding. This embedding is scored against the target user's enrollment centroid embedding using cosine similarity. If the score is equal to higher than the threshold, that attempt is authenticated.

## Setup
### Environment Setup
Firstly, I recommend setting up an virtual environment in order to not clash with your global or other packages since the project needs different versions of packages to work. 
You can either use a Conda environment (from Anaconda 3) by typing or pasting this command at the terminal:

```conda create -n myenv python=3.11```

 You can set any env name in place of "myenv"

If you prefer to use venv, please use this command to setup:

```py -3.11 -m venv .venv```

You can set any env name in place of ".venv"

Please be noted that you must install and use python 3.11 in any environment in order for the project to work.

### Dependencies
Next, you must install the necessary requirements (including packages and dependencies) using the following command:

```pip install .```

If you also want the development tools:

```pip install -e ".[dev]"```

### Running the tests
This is only for installations with development tools. Once you have completed the installation and setup, you can navigate into the tests folder by the following command:

For Windows:
```
cd src\revcaptcha\tests\
pytest test_text.py
```

All 18 tests should pass
