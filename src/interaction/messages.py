'''
CNS autobus messages
'''
# pylint: disable=too-few-public-methods

from typing import Optional, List
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


class Summarize(autobus.Event):
    ''' Summarize the current channel immediately. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    photo: bool
    max_tokens: int


class Elaborate(autobus.Event):
    ''' Continue the train of thought. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str


class Opine(autobus.Event):
    ''' Recall your opinion about entities. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    entities: List[str]


class Wikipedia(autobus.Event):
    ''' Summarize some Wikipedia pages. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    entities: List[str]


class CheckGoals(autobus.Event):
    ''' Check progress against goals. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    convo: str
    goals: List[str]


class AddGoal(autobus.Event):
    ''' Add a new goal. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    goal: str


class VibeCheck(autobus.Event):
    ''' How we feelin'? '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    convo_id: str
    room: str


class KnowledgeGraph(autobus.Event):
    '''
    The best lack all conviction, while the worst are full of passionate intensity.
    '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    convo_id: str
    convo: str


class News(autobus.Event):
    ''' What's happening in the big world? '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    url: str


class Web(autobus.Event):
    ''' Be sure to surf responsibly. '''
    service: str
    channel: str
    bot_name: str
    bot_id: str
    url: str
    reread: bool
