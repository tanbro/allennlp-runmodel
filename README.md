# allennlp-runmodel

Run a [AllenNLP] trained model, and serve it with WebAPI.

## Usage

### Run the program

Execute the program in terminator, the option `--help` will show help message:

```console
$ allennlp-runmodel --help
usage: allennlp-runmodel [-h] [--version] [--logging-config LOGGING_CONFIG]
                         [--host HOST] [--port PORT] [--path PATH]
                         [--predictor-name PREDICTOR_NAME]
                         [--cuda-device CUDA_DEVICE]
                         [--workers-type {process,thread}]
                         [--max-workers MAX_WORKERS]
                         [--num-threads NUM_THREADS]
                         archive

Run a AllenNLP trained model, and serve it with WebAPI.

positional arguments:
  archive               The archive file to load the model from.

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  --logging-config LOGGING_CONFIG, -l LOGGING_CONFIG
                        Path to logging configuration file (JSON or YAML)
                        (ref: https://docs.python.org/library/logging.config.h
                        tml#logging-config-dictschema)
  --host HOST, -s HOST  TCP/IP host or a sequence of hosts for HTTP server.
                        Default is "0.0.0.0" if port has been specified or if
                        path is not supplied.
  --port PORT, -p PORT  TCP/IP port for HTTP server. Default is 8080.
  --path PATH, -a PATH  File system path for HTTP server Unix domain socket.
                        Listening on Unix domain sockets is not supported by
                        all operating systems.
  --predictor-name PREDICTOR_NAME, -n PREDICTOR_NAME
                        Optionally specify which `Predictor` subclass;
                        otherwise, the default one for the model will be used.
  --cuda-device CUDA_DEVICE, -c CUDA_DEVICE
                        If CUDA_DEVICE is >= 0, the model will be loaded onto
                        the corresponding GPU. Otherwise it will be loaded
                        onto the CPU. (default=-1)
  --workers-type {process,thread}, -k {process,thread}
                        Sets the workers execute in thread or process.
                        (Default=process
  --max-workers MAX_WORKERS, -w MAX_WORKERS
                        Uses a pool of at most max_workers threads to execute
                        calls asynchronously. If workers_type is "process",
                        Default to the number of processors on the machine. If
                        workers_type is "thread", Default to the number of
                        processors on the machine, multiplied by 5.
  --num-threads NUM_THREADS, -t NUM_THREADS
                        Sets the number of OpenMP threads used for
                        parallelizing CPU operations. (default=4)
```

### Make prediction from HTTP client

```sh
curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"premise":"Two women are embracing while holding to go packages.","hypothesis":"The sisters are hugging goodbye while holding to go packages after just eating lunch."}' \
  http://localhost:8080/
```

------
[AllenNLP]: https://allennlp.org/
