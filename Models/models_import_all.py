"""Populate the model registry. Import order irrelevant; duplicates raise."""
from Models.mlp import mlp        # noqa: F401
from Models.cnn import cnn        # noqa: F401
from Models.rnn import rnn        # noqa: F401
from Models.ssm import s4         # noqa: F401
