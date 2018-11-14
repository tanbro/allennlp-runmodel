#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging.config
import sys
from importlib import import_module
from pathlib import Path

import torch
import yaml

from . import version


def main():
    package_name = '.'.join(version.__name__.split('.')[:-1])
    torch_num_threads: int = torch.get_num_threads()  # pylint:disable=E1101

    # arguments parsing
    parser = argparse.ArgumentParser(
        description='Run a AllenNLP trained model, and serve it with WebAPI.'
    )
    parser.add_argument('--version', action='version',
                        version=version.__version__)
    parser.add_argument(
        '--logging-config', '-l', type=Path,
        help='Path to logging configuration file (INI, JSON or YAML) '
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
             'Otherwise it will be loaded onto the CPU. '
             '(default=%(default)s)'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int,
        help='Uses a pool of at most max_workers threads to execute calls asynchronously. '
             'Default to the number of processors on the machine, multiplied by 5.'
    )
    parser.add_argument(
        '--num-threads', '-t', type=int, default=0,
        help='Sets the number of OpenMP threads used for parallelizing CPU operations. '
             f'(default={torch_num_threads})'
    )
    parser.add_argument(
        'archive', nargs=1, type=str,
        help='The archive file to load the model from.'
    )
    args = parser.parse_args()

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
        logging.basicConfig(
            format='%(asctime)-15s %(levelname).1s [%(threadName)s] %(name)s: %(message)s',
            level=logging.INFO
        )

    log = logging.getLogger(package_name)
    log.info('======== Startup ========')

    # torch's num_threads
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)  # pylint: disable=E1101
    log.info(
        'Number of OpenMP threads used for parallelizing CPU operations is %d',
        torch.get_num_threads()  # pylint: disable=E1101
    )

    # load service (import AllenNLP is VERY SLOW)
    log.info('Import service module ...')
    service = import_module('.service', package_name)
    log.info('Import service module Ok.')

    # executor
    log.info('Initial service module ...')
    service.create_executor(args.max_workers)
    log.info('Initial service module OK.')

    # model
    archive_file = args.archive[0].strip()
    log.info('Load model arhive file %r ...', archive_file)
    service.load_model(archive_file, args.cuda_device)
    log.info('Load model arhive file %r OK.', archive_file)

    if args.path:
        log.info('Start web service on %r ...', args.path)
        service.run(path=args.path)
    else:
        log.info('Start web service on "http://%s:%d" ...',
                 args.host, args.port)
        service.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
