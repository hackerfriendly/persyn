'''
CNS autobus messages
'''
# pylint: disable=too-few-public-methods

from typing import Optional, List

from sympy import O
from persyn.autobus import Event

class SendChat(Event):
    ''' Post text or images '''
    service: str
    channel: str
    msg: str
    images: Optional[list[str]] = []
    extra: Optional[str] = None


class ChatReceived(Event):
    ''' Chat was received from a service + channel '''
    service: str
    channel: str
    speaker_name: str
    msg: str
    extra: Optional[str] = None


class Idea(Event):
    ''' Inject an idea '''
    service: str
    channel: str
    idea: str
    verb: str


class Summarize(Event):
    ''' Summarize the current channel immediately. '''
    service: str
    channel: str
    convo_id: Optional[str] = None
    photo: bool
    send_chat: Optional[bool] = True
    final: Optional[bool] = False


class Elaborate(Event):
    ''' Continue the train of thought. '''
    service: str
    channel: str
    context: Optional[str] = None


class Opine(Event):
    ''' Recall your opinion about entities. '''
    service: str
    channel: str
    entities: List[str]


class Wikipedia(Event):
    ''' Use Wikipedia (via Zim) to look up entities. '''
    service: str
    channel: str
    text: str
    focus: Optional[str] = None


class CheckGoals(Event):
    ''' Check progress against goals. '''
    service: str
    channel: str
    convo: str
    goals: List[str]


class AddGoal(Event):
    ''' Add a new goal. '''
    service: str
    channel: str
    goal: str


class VibeCheck(Event):
    ''' How we feelin'? '''
    service: str
    channel: str


class FactCheck(Event):
    ''' How we thinkin'? '''
    service: str
    channel: str
    convo_id: Optional[str] = None
    room: Optional[str] = None


class KnowledgeGraph(Event):
    '''
    The best lack all conviction, while the worst are full of passionate intensity.
    '''
    service: str
    channel: str
    convo_id: str
    convo: str


class News(Event):
    ''' What's happening in the big world? '''
    service: str
    channel: str
    url: str


class Web(Event):
    ''' Be sure to surf responsibly. '''
    service: str
    channel: str
    url: str
    reread: bool


class Reflect(Event):
    ''' Reflect on the current channel. '''
    service: str
    channel: str
    send_chat: Optional[bool] = True
    convo_id: Optional[str] = None


class Photo(Event):
    ''' Generate a photo. '''
    service: str
    channel: str
    prompt: str
    size: Optional[tuple] = (1024, 1024)
