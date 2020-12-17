# MaskUp

A simple reminder to put on a mask in public spaces.

# Ensure Correct Python Version

TensorFlow requires Python3.7 or earlier, so ensure that you are currently using a compatible version of Python:

    python --version

If not, create a virtual environment by specifying an older version:

    python3.7 -m venv venv37

Navigate to the root directory and activate the virtual environment:

    source venv37/bin/activate

# Install Dependencies

After cloning the repo, install all necessary dependencies from requirements file

    pip install -r requirements.txt

# Launch Detector

**NOTE** To access native camera, must be run in MacOS Terminal app, not a third-party terminal

Audio alerts enabled:

    python detector.py audio-on

Audio alerts disabled (Default):

    python detector.py audio-off

# Stopping Program

Press Escape, or ctr-C in terminal