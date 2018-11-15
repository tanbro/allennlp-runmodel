import typing as t
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from allennlp.predictors import Predictor

logging_config: dict = None
executor: t.Union[ProcessPoolExecutor, ThreadPoolExecutor] = None
predictor: Predictor = None
