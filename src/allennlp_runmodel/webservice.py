import logging
from asyncio import get_event_loop

from aiohttp import web

from . import globvars

routes = web.RouteTableDef()  # pylint:disable=invalid-name


def predict(model_name: str, data: dict):
    predictor = globvars.predictors[model_name]
    if isinstance(data, dict):
        return predictor.predict_json(data)
    if isinstance(data, list):
        return predictor.predict_batch_json(data)
    raise ValueError('Wrong request data format')


@routes.post('/')
async def handle(request: web.Request):
    """Prediction web API handle

    .. note:: Running in main process!
    """
    logger = logging.getLogger('.'.join((__name__, 'handle')))
    loop = get_event_loop()
    rid = hex(id(request))
    peer_name = request.transport.get_extra_info('peername')
    if peer_name is not None:
        host, port = peer_name
    else:
        host, port = 'localhost', -1
    logger.debug(
        '[%s] %s==>%s:%s : %s %s ', rid, request.remote,
        host, port, request.method, request.rel_url
    )

    model_name = request.query.get('model', '')
    try:
        executor = globvars.executors[model_name]
    except KeyError:
        return web.Response(text=f'model {model_name!r} not exists', status=404)

    data = await request.json()
    logger.debug('[%s] in: %s', rid, data)

    result = await loop.run_in_executor(executor, predict, model_name, data)
    logger.debug('[%s] out: %s', rid, result)

    return web.json_response(result)


app = web.Application()  # pylint:disable=invalid-name
app.add_routes(routes)
