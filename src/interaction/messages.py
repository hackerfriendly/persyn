'''
CNS autobus messages
'''
# pylint: disable=too-few-public-methods

from typing import Optional
import autobus

class SendChat(autobus.Event):
    ''' Post text or images '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    msg: str
    images: Optional[list[str]]

class Idea(autobus.Event):
    ''' Inject an idea '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    idea: str
    verb: str
