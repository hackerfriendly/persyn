''' Color logging '''
# pylint: disable=no-self-use

import logging
import os
import sys

from click import style

__all__ = [ 'ColorLog', 'log' ]

class ColorLog():
    ''' two-color logging class. prefix is colorized, msg is not. '''
    def __init__(
            self,
            stream=sys.stderr,
            level=logging.DEBUG if 'DEBUG' in os.environ else logging.INFO,
            log_format='%(message)s'
    ):
        self.stream = stream
        self.level = level
        self.format = log_format

        logging.basicConfig(stream=self.stream, level=self.level, format=self.format)

    def debug(self, prefix, msg=''):
        ''' debug level '''
        logging.debug(' '.join([f"{style(prefix, fg='bright_cyan')}", str(msg)]))

    def info(self, prefix, msg=''):
        ''' info level '''
        logging.info(' '.join([f"{style(prefix, fg='bright_green')}", str(msg)]))

    def warning(self, prefix, msg=''):
        ''' warning level '''
        logging.warning(' '.join([f"{style(prefix, fg='bright_yellow')}", str(msg)]))

    def error(self, prefix, msg=''):
        ''' error level '''
        logging.error(' '.join([f"{style(prefix, fg='red')}", str(msg)]))

    def critical(self, prefix, msg=''):
        ''' critical level '''
        logging.critical(' '.join([f"{style(prefix, fg='bright_red')}", str(msg)]))

log = ColorLog()
