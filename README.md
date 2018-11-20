# allennlp-runmodel

Run a [AllenNLP] trained model, and serve it with WebAPI.

## Usage

### Run the program

Execute the program in terminator, the option `--help` will show help message:

```console
$ allennlp-runmodel --help
Usage: allennlp-runmodel [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

  Start a webservice for running AllenNLP models.

Options:
  -V, --version
  -h, --host TEXT                 TCP/IP host for HTTP server.  [default:
                                  localhost]
  -p, --port INTEGER              TCP/IP port for HTTP server.  [default:
                                  8000]
  -a, --path TEXT                 File system path for HTTP server Unix domain
                                  socket. Listening on Unix domain sockets is
                                  not supported by all operating systems.
  -l, --logging-config FILE       Path to logging configuration file (JSON,
                                  YAML or INI) (ref: https://docs.python.org/l
                                  ibrary/logging.config.html#logging-config-
                                  dictschema)
  -v, --logging-level [critical|fatal|error|warn|warning|info|debug|notset]
                                  Sets the logging level, only affected when
                                  `--logging-config` not specified.  [default:
                                  info]
  --help                          Show this message and exit.

Commands:
  load  Load a pre-trained AllenNLP model from it's archive file, and put
        it...

```

and

```sh
$ allennlp-runmodel load --help
Usage: allennlp-runmodel load [OPTIONS] ARCHIVE

  Load a pre-trained AllenNLP model from it's archive file, and put it into
  the webservice contrainer.

Options:
  -m, --model-name TEXT           Model name used in URL. eg: http://xxx.xxx.x
                                  xx.xxx:8000/?model=model_name
  -t, --num-threads INTEGER       Sets the number of OpenMP threads used for
                                  parallelizing CPU operations. [default: 4
                                  (on this machine)]
  -w, --max-workers INTEGER       Uses a pool of at most max_workers threads
                                  to execute calls asynchronously. [default:
                                  num_threads/cpu_count (1 on this machine)]
  -w, --worker-type [process|thread]
                                  Sets the workers execute in thread or
                                  process.  [default: process]
  -d, --cuda-device INTEGER       If CUDA_DEVICE is >= 0, the model will be
                                  loaded onto the corresponding GPU. Otherwise
                                  it will be loaded onto the CPU.  [default:
                                  -1]
  -e, --predictor-name TEXT       Optionally specify which `Predictor`
                                  subclass; otherwise, the default one for the
                                  model will be used.
  --help                          Show this message and exit.
```

`load` sub-command can be called many times to load multiple models.

eg:

```sh
allennlp-runmodel  --port 8080 load --model-name model1 /path/of/model1.tar.gz load --model-name model2 /path/of/model2.tar.gz
```

### Make prediction from HTTP client

```sh
curl \
  --header "Content-Type: application/json" \
  --request POST \
  --data '{"premise":"Two women are embracing while holding to go packages.","hypothesis":"The sisters are hugging goodbye while holding to go packages after just eating lunch."}' \
  http://localhost:8080/?model=model1
```

------
[AllenNLP]: https://allennlp.org/
