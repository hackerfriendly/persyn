'''
CNS autobus messages
'''
# pylint: disable=too-few-public-methods

from typing import Optional, List
from persyn.autobus import Event

class SendChat(Event):
    ''' Post text or images '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    msg: str
    images: Optional[list[str]]
    extra: Optional[str]


class ChatReceived(Event):
    ''' Chat was received from a service + channel '''
    service: str
    channel: str
    speaker_name: str
    msg: str
    extra: Optional[str]


class Idea(Event):
    ''' Inject an idea '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    idea: str
    verb: str


class Summarize(Event):
    ''' Summarize the current channel immediately. '''
    service: str
    channel: str
    convo_id: Optional[str]
    bot_name: str
    bot_id: str
    photo: bool
    max_tokens: int
    send_chat: Optional[bool] = True


class Elaborate(Event):
    ''' Continue the train of thought. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str


class Opine(Event):
    ''' Recall your opinion about entities. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    entities: List[str]


class Wikipedia(Event):
    ''' Summarize some Wikipedia pages. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    entities: List[str]


class CheckGoals(Event):
    ''' Check progress against goals. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    convo: str
    goals: List[str]


class AddGoal(Event):
    ''' Add a new goal. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    goal: str


class VibeCheck(Event):
    ''' How we feelin'? '''
    service: str
    channel: str
    bot_name: str
    bot_id: str


class FactCheck(Event):
    ''' How we thinkin'? '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    convo_id: Optional[str]
    room: Optional[str]


class KnowledgeGraph(Event):
    '''
    The best lack all conviction, while the worst are full of passionate intensity.
    '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    convo_id: str
    convo: str


class News(Event):
    ''' What's happening in the big world? '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    url: str


class Web(Event):
    ''' Be sure to surf responsibly. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    url: str
    reread: bool


class Reflect(Event):
    ''' Reflect on the current channel. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    send_chat: Optional[bool] = True
    convo_id: Optional[str]


class Photo(Event):
    ''' Generate a photo. '''
    service: str
    channel: str
    prompt: str
    size: Optional[tuple] = (1024, 1024)
    bot_name: str
    bot_id: str
