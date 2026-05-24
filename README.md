# An attempt to replicate the methods used in the AlphaGo paper
## Mastering the game of Go with deep neural networks and tree search
https://www.nature.com/articles/nature16961

My inspiration to develop and learn about AlphaGo came from the documentary: [AlphaGo - The Movie | Full award-winning documentary](https://www.youtube.com/watch?v=WXuK6gekU1Y). This is my first-ever neural net written and trained. I have learnt this from scratch as I implemented this.

To understand the intuition behind tree search and older methods, I followed the textbook, and reused some of the game environment from [Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go)

The following methods helped me develop an intuition around tree search:
1. MCTS
2. AlphaBeta pruning
3. Random Policy

You can find the following games to compare performances:
1. [random bot vs random bot](https://github.com/thedevwonder/re-alphago/blob/master/bot_v_bot.py)
2. [abprune bot vs random bot](https://github.com/thedevwonder/re-alphago/blob/master/abprune_v_randombot.py)
3. [mcts bot vs random bot](https://github.com/thedevwonder/re-alphago/blob/master/mcts_v_randombot.py)
4. [abprune bot vs mcts bot](https://github.com/thedevwonder/re-alphago/blob/master/abprune_v_mcts.py)

AlphaGo uses 3 training pipelines to improve its policy and value estimation.
### Supervised Learning Policy Network

I've added code to create training and test data from the KGS server games.

Steps to prepare data:
  1. Download the tar.gz game files from [KGS server](https://u-go.net/gamerecords/) and put them in data dir.
  2. Run run_dataprocessor.py to generate the dataset.

The Training Pipeline for Supervised Learning is written in the notebook [alphago.ipynb](https://github.com/thedevwonder/re-alphago/blob/master/alphago.ipynb)

### Reinforcement Learning Policy Network

Uses the REINFORCE algorithm to update the policy towards winning moves.
The Training pipeline for Reinforcement Learning of Policy Network is written in the notebook [alphago.ipynb](https://github.com/thedevwonder/re-alphago/blob/master/alphago.ipynb)

### Reinforcement Learning of Value Network

Uses regression on game state and the outcome pair to train a value function which evaluates the chances of winning the game from a given game position.
The training pipeline for Value Network is written in the notebook: [alphago.ipynb](https://github.com/thedevwonder/re-alphago/blob/master/alphago.ipynb)
