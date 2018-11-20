# -*- coding: utf-8 -*-

import json
import logging
import sys
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from functools import partial
from math import ceil
from os import cpu_count
from pathlib import Path

import click
import torch
import yaml
from aiohttp import web
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from .. import globvars, version, webservice

# pylint: disable=invalid-name,too-many-arguments,unused-argument

PACKAGE: str = '.'.join(version.__name__.split('.')[:-1])
LOGGING_CONFIG = dict(
    format='%(asctime)s %(levelname)-7s [%(process)d](%(processName)s) [%(name)s] %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)


def get_logger() -> logging.Logger:
    if __name__ == '__main__':
        return logging.getLogger(PACKAGE)
    return logging.getLogger(__name__)


def initial_logging(config_path: str = None, level_name: str = None):
    ok = False
    if config_path:
        path = Path(config_path)
        with path.open() as f:
            ext_name = path.suffix.lower()
            if ext_name == '.json':
                logging.config.dictConfig(json.load(f))
                ok = True
            elif ext_name in ['.yml', '.yaml']:
                logging.config.dictConfig(yaml.load(f))
                ok = True
            elif ext_name in ['.ini', '.conf', '.cfg']:
                logging.config.fileConfig(f)
                ok = True
            else:
                print(
                    f'Un-supported logging config file name {config_path!r}.', file=sys.stderr)
    if not ok:
        print('Can NOT make a logging config by file, default config will be used.', file=sys.stderr)
        if level_name:
            LOGGING_CONFIG['level'] = logging.getLevelName(level_name.upper())
        logging.basicConfig(**LOGGING_CONFIG)


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f'{PACKAGE} version: {version.__version__}')
    ctx.exit()


def initial_worker(cli_kdargs: dict, kdargs: dict, subproc_id: int = None):
    # logging
    if subproc_id is not None:
        initial_logging(cli_kdargs['logging_config'],
                        cli_kdargs['logging_level'])
    log = get_logger()

    model_name = kdargs['model_name']
    if model_name in globvars.predictors:
        raise RuntimeError(f'Predictor {model_name} already loaded.')

    if subproc_id is not None:
        log.info('-------- Startup(%s) --------', model_name)
    # torch threads
    num_threads = kdargs['num_threads']
    if num_threads:  # torch's num_threads
        torch.set_num_threads(num_threads)
    log.info(
        'Number of OpenMP threads used for parallelizing CPU operations is %d',
        torch.get_num_threads()
    )
    # model
    log.info('load_archive(%r, %r) ...',
             kdargs['archive'], kdargs['cuda_device'])
    archive = load_archive(kdargs['archive'], kdargs['cuda_device'])
    globvars.predictors[model_name] = Predictor.from_archive(
        archive, kdargs['predictor_name'])
    # return sub-process index
    return subproc_id


_cli_kdargs = {}


@click.group(chain=True, help='Start a webservice for running AllenNLP models.')
@click.option('--version', '-V', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True)
@click.option('--host', '-h', type=click.STRING, default='localhost', show_default=True,
              help='TCP/IP host for HTTP server.'
              )
@click.option('--port', '-p', type=click.INT,
              default='8000', show_default=True,
              help='TCP/IP port for HTTP server.'
              )
@click.option('--path', '-a', type=click.STRING,
              help='File system path for HTTP server Unix domain socket. '
              'Listening on Unix domain sockets is not supported by all operating systems.'
              )
@click.option('--logging-config', '-l', type=click.Path(exists=True, dir_okay=False),
              help='Path to logging configuration file (JSON, YAML or INI) '
              '(ref: https://docs.python.org/library/logging.config.html#logging-config-dictschema)'
              )
@click.option('--logging-level', '-v', type=click.Choice([k.lower() for k, v in logging._nameToLevel.items()], case_sensitive=False),  # pylint:disable=protected-access
              default=logging.getLevelName(logging.INFO).lower(), show_default=True,
              help='Sets the logging level, only affected when `--logging-config` not specified.'
              )
def cli(*args, **kwargs):
    initial_logging(kwargs['logging_config'], kwargs['logging_level'])
    log = get_logger()
    log.info('======== Startup ========')
    log.debug('cli arguments: %s', kwargs)
    global _cli_kdargs  # pylint:disable=global-statement
    _cli_kdargs = kwargs


@cli.resultcallback()
def after_cli(*args, **kwargs):
    log = get_logger()
    if not globvars.executors:
        msg = 'No model loaded, exiting ...'
        print(msg, file=sys.stderr)
        log.warning(msg)
        sys.exit(1)
    if kwargs['path']:
        log.info('Start webservice at unix:///%s', kwargs['path'])
        fn = partial(web.run_app, webservice.app, path=kwargs['path'])
    else:
        log.info('Start webservice at http://%s:%d',
                 kwargs['host'], kwargs['port'])
        fn = partial(web.run_app, webservice.app,
                     host=kwargs['host'], port=kwargs['port'])
    log.debug('>>> run()')
    fn()
    log.debug('<<< run()')
    log.info('======== Shutdown ========')


@cli.command('load', help='Load a pre-trained AllenNLP model from it\'s archive file, and put it into the webservice contrainer.')
@click.argument('archive', type=click.Path(exists=True, dir_okay=False))
@click.option('--model-name', '-m', type=click.STRING, default='',
              help='Model name used in URL. eg: http://xxx.xxx.xxx.xxx:8000/?model=model_name'
              )
@click.option('--num-threads', '-t', type=click.INT,
              help=f'Sets the number of OpenMP threads used for parallelizing CPU operations. '
              f'[default: {torch.get_num_threads()} (on this machine)]'
              )
@click.option('--max-workers', '-w', type=click.INT,
              help='Uses a pool of at most max_workers threads to execute calls asynchronously. '
              f'[default: num_threads/cpu_count ({ceil(cpu_count()/torch.get_num_threads())} on this machine)]'
              )
@click.option('--worker-type', '-w', type=click.Choice(['process', 'thread']), default='process', show_default=True,
              help='Sets the workers execute in thread or process.')
@click.option('--cuda-device', '-d', type=click.INT, default=-1, show_default=True,
              help='If CUDA_DEVICE is >= 0, the model will be loaded onto the corresponding GPU. '
              'Otherwise it will be loaded onto the CPU.'
              )
@click.option('--predictor-name', '-e', type=click.STRING,
              help='Optionally specify which `Predictor` subclass; '
              'otherwise, the default one for the model will be used.'
              )
def load(**kwargs):
    log = get_logger()
    log.debug('start arguments: %s', kwargs)

    model_name = kwargs['model_name']
    if model_name in globvars.executors:
        raise RuntimeError(f'Duplicated model_name {model_name!r}')

    log.info('Load model %r', model_name)

    # Create Executor, Fork if using ProcessPoolExecutor!
    max_workers = kwargs['max_workers']
    if not max_workers:
        num_threads = kwargs['num_threads']
        if num_threads is None:
            num_threads = torch.get_num_threads()
        max_workers = ceil(cpu_count() / num_threads)

    if kwargs['worker_type'] == 'process':
        log.info('Create ProcessPoolExecutor(max_workers=%d)', max_workers)
        executor = ProcessPoolExecutor(max_workers)
        try:
            for fut in as_completed([
                    executor.submit(initial_worker, _cli_kdargs, kwargs, i)
                    for i in range(max_workers)
            ]):
                retval = fut.result()
                log.info('Process[%s/%d] started.', model_name, retval)
        except Exception:
            executor.shutdown()
            raise
    else:
        log.info('Create ThreadPoolExecutor(max_workers=%d)', max_workers)
        initial_worker(_cli_kdargs, kwargs)
        executor = ThreadPoolExecutor(max_workers)

    # save the executor
    globvars.executors[model_name] = executor


if __name__ == '__main__':
    cli()
