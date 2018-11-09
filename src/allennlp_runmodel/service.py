import logging
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor


executor: ThreadPoolExecutor = None
predictor: Predictor = None
routes = web.RouteTableDef()


def load_model(archive_file: str, cuda_device: int = -1, predictor_name: str = None):
    global predictor
    archive = load_archive(archive_file, cuda_device)
    predictor = Predictor.from_archive(archive, predictor_name)


def create_executor(max_workers: int = None):
    global executor
    if max_workers:
        max_workers = int(max_workers)
        if max_workers > 0:
            executor = ThreadPoolExecutor(max_workers)


@routes.post('/')
async def handle(request: web.Request):
    log = logging.getLogger(__name__)

    data = await request.json()
    log.debug('input: %s', input)

    if isinstance(data, dict):
        func = predictor.predict_json
    elif isinstance(data, list):
        func = predictor.predict_batch_json
    else:
        raise ValueError('Wrong request data format')

    result = await request.loop.run_in_executor(executor, func, data)
    log.debug('output: %s', result)

    return web.json_response(result)


def run(**kwargs):
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, **kwargs)
