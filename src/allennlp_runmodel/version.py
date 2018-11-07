from pkg_resources import parse_version

from ._version import version as __version__

__all__ = ['__version__', 'version_info']

version_info = parse_version(__version__)
