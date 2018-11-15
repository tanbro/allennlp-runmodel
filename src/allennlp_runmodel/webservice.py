import logging
import typing as t
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from os import cpu_count

from aiohttp import web


from . import glbvars


routes = web.RouteTableDef()


@routes.post('/')
async def handle(request: web.Request):
    """Prediction web API handle
    
    .. important: It's running in the main process!
    """

    log = logging.getLogger(__name__)

    data = await request.json()
    log.debug('input: %s', input)

    # TODO: 要考虑使用子进程中的 Predictor 的情况!!!!
    if isinstance(data, dict):
        func = glbvars.predictor.predict_json
    elif isinstance(data, list):
        func = glbvars.predictor.predict_batch_json
    else:
        raise ValueError('Wrong request data format')

    result = await request.loop.run_in_executor(glbvars.executor, func, data)
    log.debug('output: %s', result)

    return web.json_response(result)


def run(**kwargs):
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, **kwargs)
