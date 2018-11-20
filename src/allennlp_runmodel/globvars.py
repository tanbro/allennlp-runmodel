import typing as t
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from allennlp.predictors import Predictor

executors: t.Dict[str, t.Union[ProcessPoolExecutor,
                               ThreadPoolExecutor]] = {}  # may NOT in sub-proc
predictors: t.Dict[str, Predictor] = {}  # may in sub-proc
