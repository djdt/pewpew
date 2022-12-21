import logging

__version__ = "1.3.3"
__loglevel__ = logging.DEBUG

logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(__loglevel__)
