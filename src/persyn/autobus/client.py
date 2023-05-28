import aioredis
import asyncio, json, logging, inspect

from .scheduler import Scheduler
from .serializer import Serializer, EncryptedSerializer

logger = logging.getLogger('autobus')

class Client:
    def __init__(self, url="redis://localhost", namespace="", shared_key=None):
        self.redis_url = url
        self.namespace = namespace

        if shared_key:
            self.serializer = EncryptedSerializer(shared_key)
        else:
            self.serializer = Serializer()

        self.listeners = {}
        self.event_types = {}
        self.channels = set()
        self.scheduler = Scheduler()

        self.output = None
        self.tasks = set()
        self.state = "stopped"
        self.state_changed = None
        self.clean_up_ready = None

    def subscribe(self, cls, fn=None):
        # If no function was passed, return a function decorator
        if not fn:
            return lambda actual_fn: self.subscribe(cls, actual_fn)
        event_type = self._register(cls)
        logging.info("Subscribing %s to %s", fn.__name__, event_type)
        listeners = self.listeners.setdefault(event_type, set())
        listeners.add(fn)
        return fn
    
    def unsubscribe(self, cls, fn):
        event_type = cls.__name__
        listeners = self.listeners.get(event_type)
        if not listeners: return
        logger.info("Attempting to unsubscribe %s from %s", fn.__name__, event_type)
        listeners.discard(fn)

    def publish(self, obj):
        event_type = self._register(obj.__class__)
        channel = self._channel(event_type)
        event = self._dump(obj)
        logger.debug("Publishing %s to %s", event_type, channel)
        if not self.output:
            raise Exception("Can't publish as autobus is not running yet")
        self.output.put_nowait((channel, event))

    def schedule(self, job, fn):
        job.do(self._run_handler, fn)

    def every(self, *args):
        return self.scheduler.every(*args)

    def _channel(self, name): 
        return ":".join(("autobus", self.namespace, name))

    def _load(self, blob):
        return self.serializer.load(blob, self.event_types)

    def _dump(self, obj):
        return self.serializer.dump(obj)

    def _register(self, cls):
        name = cls.__name__
        if name not in self.event_types:
            logger.info("Registering %s", name)
            self.event_types[name] = cls
        return name

    def _dispatch(self, event):
        event_type, obj = self._load(event)
        if not obj:
            logger.debug("Discarding unknown message: %s", event_type)
            return
        listeners = self.listeners.get(event_type, None)
        if not listeners:
            logger.debug("No listeners for event: %s", event_type)
            return
        logger.debug("Dispatching %s to %d function(s)", event_type, len(listeners))
        for listener in listeners:
            try:
                self._run_handler(listener, obj)
            except Exception as e:
                logger.exception("Listener failed")

    def _run_handler(self, handler, *args):
        logger.debug("Running handler %s", handler.__name__)
        if inspect.iscoroutinefunction(handler):
            logger.debug("%s is a coroutine; launching task", handler.__name__)
            task = asyncio.create_task(handler(*args))
            task.add_done_callback(self.clean_up_ready.put_nowait)
            self.tasks.add(task)
        else:
            handler(*args)

    async def _set_state(self, state):
        async with self.state_changed:
            logger.debug("Client shifting from %s to %s", self.state, state)
            self.state = state
            self.state_changed.notify_all()

    async def _wait_for_state(self, state):
        async with self.state_changed:
            await self.state_changed.wait_for(lambda: self.state == state)

    async def _update_subscriptions(self, pubsub):
        for event_type, listeners in self.listeners.items():
            if listeners and event_type not in self.channels:
                channel = self._channel(event_type)
                logger.info("Subscribing to pubsub channel: %s", channel)
                await pubsub.subscribe(channel)
                self.channels.add(event_type)
            elif not listeners:
                channel = self._channel(event_type)
                logger.info("Unsubscribing from pubsub channel: %s", channel)
                await pubsub.unsubscribe(channel)
                self.channels.remove(event_type)

    async def _transmit(self, redis):
        logger.debug("Ready to transmit events")
        while True:
            channel, event = await self.output.get()
            logger.debug("Publishing event to %s", channel)
            await redis.publish(channel, event)
            self.output.task_done()

    async def _receive(self, redis):
        async with redis.pubsub() as pubsub:
            await self._update_subscriptions(pubsub)
            await self._set_state("running")
            logger.debug("Ready to receive events")
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is not None:
                    logger.debug("Event received")
                    self._dispatch(message["data"])
                # This can get called multiple times in a race condition. The
                # calls are basically idempotent, but it would be a problem if
                # they weren't.
                await self._update_subscriptions(pubsub)

    async def _run_scheduled(self):
        await self._wait_for_state("running")
        logger.debug("Ready to run scheduled jobs")
        while True:
            wait = self.scheduler.idle_seconds
            if wait is None:
                wait = 15 # check every so often for new tasks, just in case
            if wait > 0:
                logger.debug("Scheduler sleeping for %0.3f seconds", wait)
                await asyncio.sleep(wait)
            self.scheduler.run_pending()

    async def _clean_up_tasks(self):
        while True:
            task = await self.clean_up_ready.get()
            self.tasks.remove(task)
            await task
            self.clean_up_ready.task_done()

    async def start(self):
        if self.tasks:
            logger.debug("autobus was already running; run() is a no-op")
            return
        self.output = asyncio.Queue()
        self.clean_up_ready = asyncio.Queue()
        self.state_changed = asyncio.Condition()
        logger.info("Starting autobus (%s)", self.redis_url)
        redis = aioredis.from_url(self.redis_url, decode_responses=True)
        self.tasks.update((
            asyncio.create_task(self._transmit(redis), name="autobus_transmit"),
            asyncio.create_task(self._receive(redis), name="autobus_receive"),
            asyncio.create_task(self._run_scheduled(), name="autobus_pending"),
            asyncio.create_task(self._clean_up_tasks(), name="autobus_cleanup")
        ))
        await self._wait_for_state("running")

    async def stop(self):
        logger.info("Stopping autobus")
        if self.output:
            await self.output.join()
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self._set_state("stopped") # this state is never actually set

    async def run(self):
        try:
            await self.start()
            await self._wait_for_state("stopping")
        finally:
            await self.stop()