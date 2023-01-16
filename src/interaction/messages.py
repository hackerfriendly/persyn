'''
CNS autobus messages
'''
import autobus
import dotwiz

class SendChat(autobus.Event):
    service: str
    channel: str
    bot_name: str
    msg: str
    config: str
