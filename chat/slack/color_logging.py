''' Color logging '''
import logging
from click import style

def debug(prefix, msg=''):
    ''' debug level '''
    logging.debug(' '.join([f"{style(prefix, fg='bright_cyan')}", str(msg)]))

def info(prefix, msg=''):
    ''' info level '''
    logging.info(' '.join([f"{style(prefix, fg='bright_green')}", str(msg)]))

def warning(prefix, msg=''):
    ''' warning level '''
    logging.warning(' '.join([f"{style(prefix, fg='bright_yellow')}", str(msg)]))

def error(prefix, msg=''):
    ''' error level '''
    logging.error(' '.join([f"{style(prefix, fg='red')}", str(msg)]))

def critical(prefix, msg=''):
    ''' critical level '''
    logging.critical(' '.join([f"{style(prefix, fg='bright_red')}", str(msg)]))
