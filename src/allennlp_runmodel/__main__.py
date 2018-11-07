#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging.config
import sys
from pathlib import Path

import yaml

from . import service
from .version import __version__


def main():
    parser = argparse.ArgumentParser(
        description='Run a AllenNLP trained model, and serve it with WebAPI.'
    )
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument(
        '--log-conf', '-l', type=Path,
        help='Path to logging configuration file (INI, JSON or YAML) (default=%(default)s)'
    )
    parser.add_argument(
        '--host', '-s', type=str,
        help='TCP/IP host or a sequence of hosts for HTTP server. '
             'Default is "0.0.0.0" if port has been specified or if path is not supplied.'
    )
    parser.add_argument(
        '--port', '-p', type=int,
        help='TCP/IP port for HTTP server. Default is 8080.'
    )
    parser.add_argument(
        '--path', '-t', type=str,
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
        'archive', nargs=1, type=str,
        help='The archive file to load the model from.'
    )

    args = parser.parse_args()

    # logging
    if args.log_conf:
        ext_name = args.log_conf.suffix.lower()
        if ext_name == '.json':
            with args.log_conf.open() as f:
                logging.config.dictConfig(json.load(f))
        elif ext_name in ['.yml', '.yaml']:
            with args.log_conf.open() as f:
                logging.config.dictConfig(yaml.load(f))
        else:
            logging.config.fileConfig(str(args.log_conf))
    else:
        print('No logging config file specified. Default config will be used.', file=sys.stderr)
        logging.basicConfig(
            format='%(asctime)-15s %(levelname).1s %(name)s: %(message)s',
            level=logging.INFO
        )

    # executor
    service.create_executor(args.max_workers)

    # model
    archive_file = args.archive[0].strip()
    service.load_model(archive_file, args.cuda_device)

    if args.path:
        service.run(path=args.path)
    else:
        service.run(host=args.host, port=args.port)


if __name__ == '__main__':
    main()
