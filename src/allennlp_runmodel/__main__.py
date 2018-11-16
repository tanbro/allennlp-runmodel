#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging.config
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor, ThreadPoolExecutor
from math import ceil
from os import cpu_count
from pathlib import Path

import torch
import yaml
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from . import globvars, version, webservice
from .settings import get_settings

PACKAGE: str = '.'.join(version.__name__.split('.')[:-1])
TORCH_NUM_THREADS: int = torch.get_num_threads()  # pylint:disable=E1101


def initial_logging(args: argparse.Namespace):
    # logging
    if args.logging_config:
        with args.logging_config.open() as f:
            ext_name = args.logging_config.suffix.lower()
            if ext_name == '.json':
                logging.config.dictConfig(json.load(f))
            elif ext_name in ['.yml', '.yaml']:
                logging.config.dictConfig(yaml.load(f))
            else:
                logging.config.fileConfig(f)
    else:
        print('No logging config file specified. Default config will be used.', file=sys.stderr)
        logging.basicConfig(**get_settings()['DEFAULT_LOGGING_CONFIG'])


def initial_process(args: argparse.Namespace, subproc_id: int = None):
    # logging
    if subproc_id is not None:
        initial_logging(args)
    log = logging.getLogger(PACKAGE)
    if subproc_id is not None:
        log.info('-------- Startup --------')
    # torch threads
    if args.num_threads > 0:  # torch's num_threads
        torch.set_num_threads(args.num_threads)  # pylint: disable=E1101
    log.info(
        'Number of OpenMP threads used for parallelizing CPU operations is %d',
        torch.get_num_threads()  # pylint: disable=E1101
    )
    # model
    if globvars.predictor:
        raise RuntimeError(f'{globvars.predictor} exists already.')
    archive_file = args.archive[0]
    log.info('load_archive(%r, %r) ...', archive_file, args.cuda_device)
    archive = load_archive(archive_file, args.cuda_device)
    globvars.predictor = Predictor.from_archive(archive, args.predictor_name)
    # return sub-process index
    if subproc_id is not None:
        return subproc_id


def main():
    # arguments parsing
    parser = argparse.ArgumentParser(
        description='Run a AllenNLP trained model, and serve it with WebAPI.'
    )
    parser.add_argument('--version', '-V', action='version', version=version.__version__)
    parser.add_argument(
        '--logging-config', '-l', type=Path,
        help='Path to logging configuration file (JSON, YAML or INI) '
             '(ref: https://docs.python.org/library/logging.config.html#logging-config-dictschema)'
    )
    parser.add_argument(
        '--host', '-s', type=str, default='0.0.0.0',
        help='TCP/IP host or a sequence of hosts for HTTP server. '
             'Default is "0.0.0.0" if port has been specified or if path is not supplied.'
    )
    parser.add_argument(
        '--port', '-p', type=int, default=8000,
        help='TCP/IP port for HTTP server. Default is 8080.'
    )
    parser.add_argument(
        '--path', '-a', type=str,
        help='File system path for HTTP server Unix domain socket. '
             'Listening on Unix domain sockets is not supported by all operating systems.'
    )
    parser.add_argument(
        '--predictor-name', '-n', type=str,
        help='Optionally specify which `Predictor` subclass; '
             'otherwise, the default one for the model will be used.'
    )
    parser.add_argument(
        '--cuda-device', '-c', type=int, default=-1,
        help='If CUDA_DEVICE is >= 0, the model will be loaded onto the corresponding GPU. '
             'Otherwise it will be loaded onto the CPU. (default=%(default)s)'
    )
    parser.add_argument(
        '--num-threads', '-t', type=int, default=0,
        help='Sets the number of OpenMP threads used for parallelizing CPU operations. '
             f'(default={TORCH_NUM_THREADS} on this machine)'
    )
    parser.add_argument(
        '--workers-type', '-k', type=str, choices=['process', 'thread'], default='process',
        help='Sets the workers execute in thread or process. (Default=%(default)s)'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int,
        help='Uses a pool of at most max_workers threads to execute calls asynchronously. '
             f'Default to num_threads/cpu_count ({ceil(cpu_count()/TORCH_NUM_THREADS)} on this machine).'
    )
    parser.add_argument(
        'archive', nargs=1, type=str,
        help='The archive file to load the model from.'
    )
    args = parser.parse_args()

    # logging in main process
    initial_logging(args)
    log = logging.getLogger(PACKAGE)
    log.info('======== Startup ========')

    # Create Executor, Fork if using ProcessPoolExecutor!
    max_workers = args.max_workers
    if not max_workers:
        if args.num_threads:
            num_threads = args.num_threads
        else:
            num_threads = TORCH_NUM_THREADS
        max_workers = ceil(cpu_count() / num_threads)
    if args.workers_type == 'process':
        log.info('Create ProcessPoolExecutor(max_workers=%d)', max_workers)
        globvars.executor = ProcessPoolExecutor(max_workers)
        for fut in as_completed([
            globvars.executor.submit(initial_process, args, i)
            for i in range(max_workers)
        ]):
            log.info('Process-%d started.', fut.result() + 1)
    else:  # thread worker
        log.info('Create ThreadPoolExecutor(max_workers=%d)', max_workers)
        globvars.executor = ThreadPoolExecutor(args.max_workers)
        initial_process(args)

    with globvars.executor:
        if args.path:
            log.info('Start web-service on %r ...', args.path)
            webservice.run(path=args.path)
        else:
            log.info('Start web-service on "http://%s:%d" ...',
                     args.host, args.port)
            webservice.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
