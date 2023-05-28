import asyncio

from .client import Client
from .event import Event

client = Client()

def subscribe(event_cls):
    return client.subscribe(event_cls)

def unsubscribe(event_cls, fn):
    client.unsubscribe(event_cls, fn)

def schedule(job):
    def schedule_decorator(fn):
        client.schedule(job, fn)
        return fn
    return schedule_decorator

def every(*args):
    return client.every(*args)

def publish(event):
    client.publish(event)

def start(namespace="", url="redis://localhost"):
    client.namespace = namespace
    client.redis_url = url
    return client.start()

def stop():
    return client.stop()

def run(namespace="", url="redis://localhost"):
    client.namespace = namespace
    client.redis_url = url
    asyncio.run(client.run())
