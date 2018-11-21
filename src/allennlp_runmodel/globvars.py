from concurrent.futures import Executor
from typing import Dict

from allennlp.predictors import Predictor

executors: Dict[str, Executor] = {}  # may NOT in sub-proc
predictors: Dict[str, Predictor] = {}  # may in sub-proc
