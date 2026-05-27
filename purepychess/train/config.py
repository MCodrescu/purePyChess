"""Training-time configuration for purePyChess.

Change TARGET_ELO here to adjust the quality filter applied in both
preprocess.py and pretrain.py.
"""

TARGET_ELO            = 1800   # minimum average Elo to include a game
MIN_TIME_CONTROL      = 180    # minimum base time in seconds
REPLAY_BUFFER_SIZE    = 500_000
NUM_SELFPLAY_WORKERS  = 4
