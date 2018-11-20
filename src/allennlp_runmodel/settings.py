import logging
import sys

_settings = dict(
    DEFAULT_LOGGING_CONFIG={
        'format': '%(asctime)s %(levelname)-7s [%(process)d](%(processName)s) [%(name)s] %(message)s',
        'level': logging.INFO,
        'stream': sys.stdout,
    },
)


def get_settings():
    return _settings


def set_setting(k, v):
    _settings[k] = v
