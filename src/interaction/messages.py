'''
CNS autobus messages
'''
from typing import Optional
import autobus

class SendChat(autobus.Event):
    service: str
    channel: str
    bot_name: str
    msg: str
    images: Optional[list[str]]
