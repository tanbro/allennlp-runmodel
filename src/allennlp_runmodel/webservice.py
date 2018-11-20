import logging

from aiohttp import web

from . import globvars

routes = web.RouteTableDef()


def predict(data):
    if isinstance(data, dict):
        return globvars.predictor.predict_json(data)
    elif isinstance(data, list):
        return globvars.predictor.predict_batch_json(data)
    else:
        raise ValueError('Wrong request data format')


@routes.post('/')
async def handle(request: web.Request):
    """Prediction web API handle
    
    .. note:: Running in main process!
    """
    log = logging.getLogger(__name__)
    rid = hex(id(request))

    data = await request.json()
    log.debug('[%s] in: %s', rid, data)

    result = await request.loop.run_in_executor(globvars.executor, predict, data)
    log.debug('[%s] out: %s', rid, result)

    return web.json_response(result)


def run(**kwargs):
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, **kwargs)
