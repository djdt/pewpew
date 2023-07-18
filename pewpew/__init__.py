import logging

__version__ = "1.4.0"
__loglevel__ = logging.DEBUG

logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(__loglevel__)
