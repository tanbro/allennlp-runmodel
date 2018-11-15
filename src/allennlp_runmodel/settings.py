import logging

_settings = dict(
    DEFAULT_LOGGING_CONFIG={
        'format': '%(asctime)-15s %(levelname)s [%(process)d](%(processName)s) %(name)s %(message)s',
        'level': logging.INFO,
    },
)


def get_settings():
    return _settings


def set_setting(k, v):
    _settings[k] = v
