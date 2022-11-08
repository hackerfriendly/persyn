'''
reminders.py

Manage background threads to take some future action.
'''
# pylint: disable=import-error, wrong-import-position
import sys
import asyncio
import threading as th

from pathlib import Path

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../').resolve()))

from utils.color_logging import log

__all__ = [ 'Reminders', 'reminders' ]

class Reminders():
    ''' Container class for managing reminder threads '''
    def __init__(self):
        ''' da setup '''
        self.reminders = {}

    def new_channel(self, channel):
        ''' Initialize a new channel. '''
        self.reminders[channel] = {
            'default': th.Timer(0, log.warning, ["New default reminder for channel:", channel])
        }

    def add(self, channel, when, func, args=None, name='default'):
        ''' Add a reminder '''
        if channel not in self.reminders:
            self.new_channel(channel)

        if name in self.reminders[channel]:
            # one at a time
            self.reminders[channel][name].cancel()

        self.reminders[channel][name] = th.Timer(when, func, args)
        self.reminders[channel][name].start()

    def cancel(self, channel, name='default'):
        ''' Cancel a reminder '''
        if channel not in self.reminders:
            self.new_channel(channel)
            return

        self.reminders[channel][name].cancel()


class AsyncReminders():
    ''' Container class for managing reminder coroutines '''
    def __init__(self):
        ''' da setup '''
        self.reminders = {}

    def new_channel(self, channel):
        ''' Initialize a new channel. '''
        self.reminders[channel] = {
            'default': asyncio.create_task(self.wait_for_it(0, log.warning, f"New default reminder for channel: {channel}"))
        }

    async def wait_for_it(self, when, func, args):
        ''' Wait then execute '''
        await asyncio.sleep(when)
        await func(args)

    def add(self, channel, when, func, args=None, name='default'):
        ''' Add a reminder '''
        if channel not in self.reminders:
            self.new_channel(channel)

        if name in self.reminders[channel]:
            # one at a time
            self.reminders[channel][name].cancel()

        self.reminders[channel][name] = asyncio.create_task(self.wait_for_it(when, func, args))

    def cancel(self, channel, name='default'):
        ''' Cancel a reminder '''
        if channel not in self.reminders:
            self.new_channel(channel)
            return

        self.reminders[channel][name].cancel()

async_reminders = AsyncReminders()
reminders = Reminders()
