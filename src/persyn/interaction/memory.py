''' memory.py: long and short term memory by Redis. '''
# pylint: disable=invalid-name, no-name-in-module, abstract-method
import uuid

from dataclasses import dataclass
from typing import Union

import ulid

import redis
from redis.commands.search.query import Query

from langchain.memory import (
    CombinedMemory,
    ConversationSummaryBufferMemory,
    ConversationKGMemory
)

from langchain.vectorstores.redis import Redis

from persyn.interaction.chrono import elapsed, get_cur_ts
from persyn.interaction.completion import LanguageModel
from persyn.utils.color_logging import ColorLog
from persyn.utils.config import PersynConfig

log = ColorLog()

@dataclass
class Convo:
    ''' Container class for conversations. '''
    service: str
    channel: str

    def __post_init__(self):
        self.convo_id = ulid.ULID()
        self.id = str(self.convo_id)
        self.memories = {}

    def __repr__(self):
        return f"service='{self.service}', channel='{self.channel}', id='{self.id}'"

    def __str__(self):
        return f"{self.service}|{self.channel}|{self.id}"

@dataclass
class Recall:
    '''
    Total Recall

    Track conversations. If the conversation_interval has expired, start a new one.

    Also includes helpers for accessing the knowledge graph.
    '''
    config: PersynConfig

    def __post_init__(self, conversation_interval=None):

        self.bot_name = self.config.id.name
        self.bot_id = uuid.UUID(self.config.id.guid)
        self.bot_ulid = ulid.ULID().from_uuid(self.bot_id)

        self.conversation_interval = conversation_interval or self.config.memory.conversation_interval
        self.max_summary_size = self.config.memory.max_summary_size

        self.redis = redis.from_url(self.config.memory.redis)

        self.lm = LanguageModel(self.config)

        # indices
        pre = f'persyn2:{self.bot_id}'
        self.convo_prefix = f'{pre}:convo'
        self.summary_prefix = f'{pre}:summary'
        self.opinion_prefix = f'{pre}:opinion'
        self.goal_prefix = f'{pre}:goal'
        self.news_prefix = f'{pre}:news'

        # sets
        self.active_convos_prefix = f"{pre}:active_convos"

        # Convenience ids. Useful when browsing with RedisInsight.
        self.redis.hset(f"{pre}:whoami", "bot_name", self.bot_name)
        self.redis.hset(f"{pre}:whoami", "bot_id", str(self.bot_id))
        self.redis.hset(f"{pre}:whoami", "bot_ulid", str(self.bot_ulid))

        # Container for current conversations
        self.convos = {}

    def list_convos(self):
        ''' Return the set of all active convos for all services + channels '''
        # TODO: filter for active convos only
        return self.convos.keys()

    # def judge(self, service, channel, topic, opinion, convo_id):
    #     ''' Judge not, lest ye be judged '''
    #     log.warning(f"👨‍⚖️ judging {topic}")
    #     return self.save_opinion(service, channel, topic, opinion, convo_id)

    # def surmise(self, service, channel, topic, size=10):
    #     ''' Everyone's got an opinion '''
    #     log.warning(f"📌 opinion on {topic}")
    #     return self.lookup_opinions(service, channel, topic, size)

    def convo(self, service, channel, convo_id=None, feels=False, verb=None, raw=False, size=1000):
        '''
        Return an entire convo.

        If convo_id is None, use the current convo (if any).

        If feels is True, also include feelings.

        If verb is not None, filter to only include messages with a matching verb.

        If raw is true, return convo objects. Otherwise return a list of strings.
        '''
        raise NotImplementedError
        # if convo_id is None:
        #     lm = self.get_last_message(service, channel)
        #     if not lm:
        #         return []
        #     convo_id = lm.convo_id

        # if verb:
        #     query = (
        #         Query(
        #             """(@service:{$service}) (@channel:{$channel}) (@convo_id:{$convo_id}) (@verb:{$verb})"""
        #         )
        #         .return_fields(
        #             "speaker_name",
        #             "msg",
        #             "verb",
        #             "convo_id"
        #         )
        #         .sort_by("convo_id", asc=True)
        #         .paging(0, size)
        #         .dialect(2)
        #     )
        #     query_params = {"service": service, "channel": channel, "convo_id": convo_id, "verb": verb}
        # else:
        #     query = (
        #         Query(
        #             """(@service:{$service}) (@channel:{$channel}) (@convo_id:{$convo_id})"""
        #         )
        #         .return_fields(
        #             "speaker_name",
        #             "msg",
        #             "verb",
        #             "convo_id"
        #         )
        #         .sort_by("convo_id", asc=True)
        #         .paging(0, size)
        #         .dialect(2)
        #     )
        #     query_params = {"service": service, "channel": channel, "convo_id": convo_id}

        # ret = []
        # realization = None
        # for msg in self.redis.ft(self.convo_prefix).search(query, query_params).docs:
        #     if not getattr(msg, 'verb', None):
        #         continue

        #     if msg.verb == 'new_convo':
        #         continue
        #     if msg.verb == 'realizes':
        #         realization = f"{msg.speaker_name} {msg.verb}: {msg.msg}"
        #         continue

        #     if msg.verb == 'dialog':
        #         if raw:
        #             ret.append(msg)
        #         else:
        #             ret.append(f"{msg.speaker_name}: {msg.msg}")
        #     elif feels or msg.verb != 'feels':
        #         if raw:
        #             ret.append(msg)
        #         else:
        #             ret.append(f"{msg.speaker_name} {msg.verb}: {msg.msg}")

        # if realization:
        #     ret.append(realization)

        # return ret

    def feels(self, convo_id):
        '''
        Return the last known feels for this convo_id.
        '''
        raise NotImplementedError
        # query = (
        #     Query(
        #         """(@convo_id:{$convo_id}) (@verb:{feels})"""
        #     )
        #     .return_fields(
        #         "msg"
        #     )
        #     .sort_by("convo_id", asc=False)
        #     .paging(0, 1)
        #     .dialect(2)
        # )
        # query_params = {"convo_id": convo_id}

        # ret = self.redis.ft(self.convo_prefix).search(query, query_params).docs
        # if ret:
        #     return ret[0].msg

        # return "nothing in particular"

    def create_lc_memories(self) -> dict:
        ''' Create a fresh set of langchain memories. '''
        summary_memory =  ConversationSummaryBufferMemory(
            llm=self.lm.summary_llm,
            max_token_limit=self.max_summary_size,
            return_messages=False,
            input_key="input",
            ai_prefix=self.config.id.name
            # human_prefix is set on first use
        )

        kg_memory = ConversationKGMemory(
            llm=self.lm.summary_llm,
            input_key="input",
            memory_key="kg",
            ai_prefix=self.config.id.name,
            # human_prefix is set on first use
        )

        rds = Redis(
            redis_url=self.config.memory.redis,
            index_name=f"persyn-{self.config.id.guid}-convo",
            embedding=self.lm.embeddings,
            key_prefix=self.convo_prefix
        )

        combined_memory = CombinedMemory(memories=[summary_memory, kg_memory])

        return { 'combined': combined_memory, 'redis': rds }


    def new_convo(self, service, channel) -> Convo:
        ''' Start a new conversation. '''
        convo = Convo(
            service=service,
            channel=channel
        )
        convo.memories=self.create_lc_memories()

        log.warning("⚠️  New convo:", convo)
        self.convos[str(convo)] = convo

        return convo

    def current_convo_id(self, service, channel) -> Union[str, None]:
        ''' Return the current convo_id for service and channel (if any) '''
        convo_ids = sorted([k for k in self.convos if k.startswith(f'{service}|{channel}|')])
        # TODO: check for expiration
        if convo_ids:
            return convo_ids[-1].split('|', maxsplit=2)[-1]
        return None

    def get_convo(self, service, channel, convo_id=None) -> Convo:
        ''' If convo_id exists, fetch it. If not, start a new convo. '''
        log.warning(f"get_convo(): {service} | {channel} | {convo_id}")
        if convo_id is None:
            convo_id = self.current_convo_id(service, channel)

        log.warning(f"Going with convo_id: {convo_id}")
        convo_key = f"{service}|{channel}|{convo_id}"

        log.warning(self.convos)
        if convo_id is None or convo_key not in self.convos:
            log.warning(f"Couldn't find {convo_key}, starting fresh")
            return self.new_convo(service, channel)

        return self.convos[convo_key]

    def expired(self, service, channel):
        ''' True if time elapsed since the last convo line is > conversation_interval, else False '''
        return elapsed(self.get_last_timestamp(service, channel), get_cur_ts()) > self.conversation_interval

    def convo_id(self, service, channel):
        ''' Return the current convo id. Make a new convo if needed. Old convos are expired in cns.py. '''

        ret = self.get_last_message(service, channel)
        if not ret:
            return None

        if self.expired(service, channel):
            return self.new_convo(service, channel)

        return ret.convo_id

    @staticmethod
    def entity_id_to_epoch(entity_id):
        ''' Extract the epoch seconds from a ULID '''
        if entity_id is None:
            return 0

        if isinstance(entity_id, str):
            return ulid.ULID().from_str(entity_id).timestamp

        return entity_id.timestamp

    def entity_id_to_timestamp(self, entity_id):
        ''' Extract the timestamp from a ULID '''
        return get_cur_ts(self.entity_id_to_epoch(entity_id))

    def get_last_timestamp(self, service, channel):
        '''
        Get the timestamp of the last message, or the current ts if there is none.
        '''
        msg = self.get_last_message(service, channel)
        if msg:
            try:
                return get_cur_ts(epoch=ulid.ULID().from_str(msg.pk).timestamp)
            except AttributeError:
                log.warning("‼️ entity_id_to_epoch(): no pk for msg:", msg)

        return get_cur_ts()


    def get_last_message(self, service, channel):
        ''' Return the last message seen on this channel '''
        query = (
            Query(
                """(@service:{$service}) (@channel:{$channel})"""
            )
            .return_fields(
                "service",
                "channel",
                "convo_id",
                "speaker_name",
                "msg",
                "verb",
                "pk",
                "id",
            )
            .sort_by("convo_id", asc=False)
            .paging(0, 1)
            .dialect(2)
        )
        query_params = {"service": service, "channel": channel}

        ret = self.redis.ft(self.convo_prefix).search(query, query_params).docs
        if not ret:
            return None

        return ret[0]


    def get_summary_by_id(self, convo_id):
        ''' Return the last summary for this convo_id '''
        query = Query("(@convo_id:{$convo_id})").sort_by("convo_id", asc=False).paging(0, 1).dialect(2)
        query_params = {"convo_id": convo_id}
        try:
            return self.redis.ft(self.summary_prefix).search(query, query_params).docs[0]
        except IndexError:
            return None

    @staticmethod
    def entity_key(service, channel, name):
        ''' Unique string for each service + channel + name '''
        return f"{str(service).strip()}|{str(channel).strip()}|{str(name).strip()}"

    @staticmethod
    def uuid_to_entity(the_uuid):
        ''' Return the equivalent short ID (str) for a uuid (str) '''
        return str(ulid.ULID().from_uuid(the_uuid))

    @staticmethod
    def entity_to_uuid(entity_id):
        ''' Return the equivalent UUID (str) for a short ID (str) '''
        return ulid.ULID().from_str(str(entity_id)).to_uuid()

    def name_to_entity_id(self, service, channel, name):
        ''' One distinct short UUID per bot_id + service + channel + name '''
        return self.uuid_to_entity(uuid.uuid5(self.bot_id, self.entity_key(service, channel, name)))
