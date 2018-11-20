import typing as t
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from allennlp.predictors import Predictor

executor: t.Union[ProcessPoolExecutor, ThreadPoolExecutor] = None
predictor: Predictor = None
