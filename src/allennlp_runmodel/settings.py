import logging

_settings = dict(
    DEFAULT_LOGGING_CONFIG={
        'format': '%(asctime)-15s %(levelname)s - %(processName)s(%(process)d) - %(threadName)s - %(name)s - %(message)s',
        'level': logging.INFO,
    },
)


def get_settings():
    return _settings


def set_settting(k, v):
    _settings[k] = v
