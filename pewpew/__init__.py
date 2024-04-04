import logging

__loglevel__ = logging.DEBUG

logging.captureWarnings(True)
log = logging.getLogger()
log.setLevel(__loglevel__)
