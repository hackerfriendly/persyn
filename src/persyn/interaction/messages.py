'''
CNS autobus messages
'''
# pylint: disable=too-few-public-methods

from typing import Optional, List

from persyn.autobus import Event

# FIXME: Constantly providing service + channel + convo_id is redundant and unwieldy. Rework messages to only require convo_id.

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
    convo_id: Optional[str] = None
    context: Optional[str] = None

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
    size: Optional[tuple[int, int]] = (1024, 1024)
