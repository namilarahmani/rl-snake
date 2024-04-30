# rl-final-project

## Installing necessary packages
Run `pip install -r requirements.txt` to install all the necessary packages for the code. It is recommended to use a virtual environment for this. 

## Running the Code
We modified existing baseline code to implement our algorithms. In `snakeClass.py`, there is a dictionary of parameters to modify for the various algorithms as follows:
- Baseline (random): Set `params['mode']` to 'baseline' and `params['DDQN']` to False
- Linear Q Learning: Set `params['mode']` to 'linear-q' and `params['DDQN']` to False
- DQN: Set `params['mode']` to 'epsilon-greedy' and `params['DDQN']` to False
- Double-DQN (random): Set `params['mode']` to 'epsilon-greedy' and `params['DDQN']` to True

To start the simulation, run the `snakeClass.py` file using Python (`python3 snakeClass.py`). 
