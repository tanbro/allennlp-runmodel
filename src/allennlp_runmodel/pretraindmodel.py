import logging
from concurrent.futures import ThreadPoolExecutor

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

_predictor: Predictor = None
_executor: ThreadPoolExecutor = None


def initial(archive_file: str, cuda_device: int = -1, max_workers: int = None, predictor_name: str = None):
    log = logging.getLogger(__name__)
    global _predictor, _executor
    if max_workers:
        log.info('create ThreadPoolExecutor(max_workers=%d)', max_workers)
        _executor = ThreadPoolExecutor(max_workers)
    log.info('load_archive(%s)', archive_file)
    _archive = load_archive(archive_file, cuda_device)
    _predictor = Predictor.from_archive(_archive, predictor_name)


def get_predictor():
    return _predictor


def get_executor():
    return _executor
