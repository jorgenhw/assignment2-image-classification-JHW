# install requirements.txt for project
# Usage: source setup.sh

# create virtual environment
python3 -m venv assignment_2_VISUAL_env

# activate virtual environment
source ./assignment_2_VISUAL_env/bin/activate

# Install requirements
python3 -m pip install --upgrade pip # upgrade pip
pip3 install -r requirements.txt

# run the code
python3 main.py

# Extra: Type ```python3 main.py --help``` in terminal to see the help message for the adjustable parameters

# Deactivate virtual environment
deactivate