import asyncio
import logging

from aiohttp import web

from .pretraindmodel import get_executor, get_predictor

routes = web.RouteTableDef()


@routes.post('/predict')
async def predict(request: web.Request):
    log = logging.getLogger(__name__)
    loop = asyncio.get_event_loop()

    data = await request.json()
    log.debug('predict: %s', input)

    result = await loop.run_in_executor(
        get_executor(),
        get_predictor().predict_json,
        data
    )
    return web.json_response(result)


@routes.post('/predict_many')
async def predict_many(request: web.Request):
    log = logging.getLogger(__name__)
    loop = asyncio.get_event_loop()

    data = await request.json()
    log.debug('predict_many: %s', data)

    result = await loop.run_in_executor(
        get_executor(),
        get_predictor().predict_batch_json,
        data
    )
    return web.json_response(result)


def run(**kwargs):
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, **kwargs)
