# rl-final-project

Todo:

- [x] Get existing implementation of snake game working
- [x] Implement baseline algorithm (randomly choosing from feasible actions)
- [ ] Implement tile coding for grid representation
- [ ] Implement linear Q-learning and collect results
- [ ] Implement deep Q-learning and collect results
- [ ] Implement PPO and collect results
- [ ] Repeat all of the above experiments with variation of extra apples on screen
- [ ] Maybe implement as POMDP

Notes for us:
- looks like tabular RL is not going to work in this context (too complicated)
- the implemented snake game is in `snake-ga` and we just need to run `python3 snakeClass.py` to start the game
- I implemented the baseline (just randomly picking an action) in `basline.py`; this function is imported in `snakeClass.py`. We can follow a similar set up for the rest of the methods where we just implement the functions by taking in the old state and returning a single number denoting the action
