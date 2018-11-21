# -*- coding: utf-8 -*-

import asyncio
import json
import logging.config
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
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

_cli_kdargs = {}
_logging_initialed = False


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
        level_name = level_name.strip().upper()
        if level_name:
            LOGGING_CONFIG['level'] = logging.getLevelName(level_name)
        logging.basicConfig(**LOGGING_CONFIG)


def initial_worker(cli_kdargs: dict, kdargs: dict, subproc_id: int = None):
    # logging
    if subproc_id is not None:
        initial_logging(cli_kdargs['logging_config'], cli_kdargs['logging_level'])
    log = get_logger()

    model_name = kdargs['model_name']
    if model_name in globvars.predictors:
        raise RuntimeError(f'Predictor {model_name} already loaded.')

    if subproc_id is not None:
        log.info('-------- Startup(%s[%d]) --------', model_name, subproc_id)
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


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo(f'{PACKAGE} version: {version.__version__}')
    ctx.exit()


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
@click.option('--logging-level', '-v',
              type=click.Choice(
                  ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'WARN', 'INFO', 'DEBUG'],
                  case_sensitive=False
              ),
              default=logging.getLevelName(logging.INFO).lower(), show_default=True,
              help='Sets the logging level, only affected when `--logging-config` not specified.'
              )
def cli(*args, **kwargs):
    global _cli_kdargs  # pylint:disable=global-statement
    _cli_kdargs = kwargs


@cli.resultcallback()
def after_cli(*args, **kwargs):
    log = get_logger()
    loop = asyncio.get_event_loop()
    if not globvars.executors:
        msg = 'No model loaded, exiting ...'
        print(msg, file=sys.stderr)
        log.warning(msg)
        sys.exit(1)

    log.debug('setup webservice runner')
    runner = web.AppRunner(webservice.app)
    loop.run_until_complete(runner.setup())

    if kwargs['path']:
        log.info('create webservice site at unix:///%s', kwargs['path'])
        site = web.SockSite(runner, kwargs['path'])
    else:
        log.info('create webservice site at http://%s:%d', kwargs['host'], kwargs['port'])
        site = web.TCPSite(runner, kwargs['host'], kwargs['port'])

    log.debug('start webservice site')
    loop.run_until_complete(site.start())

    log.debug('>>> run()')
    try:
        loop.run_forever()
    except Exception:
        log.exception('Un-caught exception:')
        raise
    finally:
        log.debug('<<< run()')
        log.info('======== Shutdown ========')


@cli.command('load',
             help='Load a pre-trained AllenNLP model from it\'s archive file, '
                  'and put it into the webservice container.'
             )
@click.argument('archive', type=click.Path(exists=True, dir_okay=False))
@click.option('--model-name', '-m', type=click.STRING, default='', show_default=True,
              help='Model name used in URL. eg: `http://host:80/?model=name` '
                   'Empty model name by default.'
              )
@click.option('--num-threads', '-t', type=click.INT,
              help='Sets the number of OpenMP threads used for paralleling CPU operations. '
                   f'[default: {torch.get_num_threads()} (on this machine)]'
              )
@click.option('--max-workers', '-w', type=click.INT,
              help='Uses a pool of at most max_workers threads to execute calls asynchronously. '
                   f'[default: num_threads/cpu_count ({ceil(cpu_count()/torch.get_num_threads())} on this machine)]'
              )
@click.option('--worker-type', '-w', type=click.Choice(['PROCESS', 'THREAD'], case_sensitive=False),
              default='process', show_default=True,
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
    initial_logging(_cli_kdargs['logging_config'], _cli_kdargs['logging_level'])
    log = get_logger()
    log.debug('load arguments: %s', kwargs)

    if not globvars.executors:  # First load
        log.info('======== Startup ========')

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

    worker_type = kwargs['worker_type'].strip().upper()
    if worker_type == 'PROCESS':
        log.info('Create ProcessPoolExecutor(max_workers=%d)', max_workers)
        executor = ProcessPoolExecutor(max_workers)
        try:
            # pylint:disable=bad-continuation
            for fut in as_completed([
                executor.submit(initial_worker, _cli_kdargs, kwargs, i)
                for i in range(max_workers)
            ]):
                try:
                    subproc_id = fut.result()
                except Exception:
                    log.exception('[%s]Process failed on initializing.', model_name)
                    raise
                else:
                    log.info('[%s]Process[%d] initialized.', model_name, subproc_id)
        except Exception:
            executor.shutdown()
            raise
    elif worker_type == 'THREAD':
        log.info('Create ThreadPoolExecutor(max_workers=%d)', max_workers)
        initial_worker(_cli_kdargs, kwargs)
        executor = ThreadPoolExecutor(max_workers)
    else:
        raise ValueError(f'Un-supported worker-type {worker_type!r}')

    # save the executor
    globvars.executors[model_name] = executor


if __name__ == '__main__':
    cli()
