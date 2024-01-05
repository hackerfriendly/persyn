''' memory.py: long and short term memory by Redis. '''
# pylint: disable=invalid-name, no-name-in-module, abstract-method
import uuid

from dataclasses import dataclass
from typing import Union, List, Any, Optional

import ulid

import redis

from langchain.memory import (
    CombinedMemory,
    ConversationSummaryBufferMemory,
    ConversationKGMemory
)

from langchain.vectorstores.redis import Redis
from redis.commands.search.query import Query

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
    id: Optional[str] = None

    def __post_init__(self):
        ''' validate id '''
        if self.id is None:
            self.id = str(ulid.ULID())
        else:
            self.id = str(ulid.ULID().from_str(self.id))

        self.memories = {}              # langchain working memories
        self.visited = set([self.id])   # other convo_ids and ideas we have visited

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

        self.conversation_interval = conversation_interval or self.config.memory.conversation_interval
        self.max_summary_size = self.config.memory.max_summary_size

        self.redis = redis.from_url(self.config.memory.redis)

        self.lm = LanguageModel(self.config)

        # indices
        pre = f'persyn2:{self.bot_id}'
        self.convo_prefix = f'{pre}:convo'
        self.opinion_prefix = f'{pre}:opinion'
        self.goal_prefix = f'{pre}:goal'
        self.news_prefix = f'{pre}:news'

        self.convo_schema = {
            "tag": [
                {"name":"service"},
                {"name":"channel"},
                {"name":"expired"},
                {"name":"role"},
                {"name":"speaker_name"},
                {"name":"verb"},
                {"name":"initiator"}
            ],
            "text": [
                {"name":"convo_id"},
                {"name":"feels"}
            ]

        }

        # Convenience ids. Useful when browsing with RedisInsight.
        self.redis.hset(f"{pre}:whoami", "bot_name", self.bot_name)
        self.redis.hset(f"{pre}:whoami", "bot_id", str(self.bot_id))

        # Container for current conversations
        self.convos = {}

        # Create indices
        for cmd in [
            f"FT.CREATE {self.convo_prefix} on HASH PREFIX 1 {self.convo_prefix}: SCHEMA service TAG channel TAG expired TAG role TAG speaker_name TAG initiator TAG verb TAG content TEXT convo_id TEXT feels TEXT content_vector VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE",
            f"FT.CREATE {self.opinion_prefix} on HASH PREFIX 1 {self.opinion_prefix}: SCHEMA service TAG channel TAG opinion TEXT topic TEXT convo_id TAG",
            f"FT.CREATE {self.goal_prefix} on HASH PREFIX 1 {self.goal_prefix}: SCHEMA service TAG channel TAG goal TEXT",
            f"FT.CREATE {self.news_prefix} on HASH PREFIX 1 {self.news_prefix}: SCHEMA service TAG channel TAG url TEXT title TEXT",
        ]:
            try:
                self.redis.execute_command(cmd)
                log.warning("Creating index:", cmd.split(' ')[1])
            except redis.exceptions.ResponseError as err:
                if str(err) == "Index already exists":
                    log.debug(f"{err}:", cmd.split(' ')[1])
                else:
                    log.error(f"{err}:", cmd.split(' ')[1])
                continue

    # def judge(self, service, channel, topic, opinion, convo_id):
    #     ''' Judge not, lest ye be judged '''
    #     log.warning(f"👨‍⚖️ judging {topic}")
    #     return self.save_opinion(service, channel, topic, opinion, convo_id)

    # def surmise(self, service, channel, topic, size=10):
    #     ''' Everyone's got an opinion '''
    #     log.warning(f"📌 opinion on {topic}")
    #     return self.lookup_opinions(service, channel, topic, size)


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
            index_name=self.convo_prefix,
            index_schema=self.convo_schema,
            embedding=self.lm.embeddings, # FIXME: 8192 context limit?
            key_prefix=self.convo_prefix,
        )

        combined_memory = CombinedMemory(memories=[summary_memory, kg_memory])

        return {
                'combined': combined_memory,
                'summary': summary_memory,
                'kg': kg_memory,
                'redis': rds
        }

    def decode_dict(self, d) -> dict:
        ''' Decode a dict fetched from Redis '''
        ret = {}
        for (k, v) in d.items():
            ret[k.decode()] = v.decode()
        return ret

    def set_convo_meta(self, convo_id, k, v) -> None:
        ''' Set metadata for a conversation '''
        self.redis.hset(f"{self.convo_prefix}:{convo_id}:meta", k, v)

    def fetch_convo_meta(self, convo_id, k=None) -> Any:
        ''' Fetch a metadata key from a conversation in Redis. If k is not provided, return all metadata. '''
        try:
            if k is None:
                return self.decode_dict(self.redis.hscan(f'{self.convo_prefix}:{convo_id}:meta')[1].items())
            return self.redis.hget(f"{self.convo_prefix}:{convo_id}:meta", k).decode()
        except AttributeError:
            return None

    def fetch_summary(self, convo_id) -> Union[str, None]:
        ''' Fetch a conversation summary. '''
        ret = self.redis.hget(f"{self.convo_prefix}:{convo_id}:summary", "content")
        if ret:
            return ret.decode()
        log.debug("👎 No summary found for:", convo_id)
        return None

    def new_convo(self, service, channel, speaker_name, convo_id=None) -> Convo:
        ''' Start a new conversation. If convo_id is not supplied, generate a new one. '''
        convo = Convo(
            service=service,
            channel=channel,
            id=convo_id
        )

        if convo_id:
            log.warning("👉 Continuing convo:", convo)
        else:
            log.warning("⚠️  New convo:", convo)

        self.set_convo_meta(convo.id, "service", service)
        self.set_convo_meta(convo.id, "channel", channel)
        self.set_convo_meta(convo.id, "initiator", speaker_name)
        self.set_convo_meta(convo.id, "expired", "False")

        convo.memories=self.create_lc_memories()
        convo.memories['summary'].human_prefix = speaker_name
        convo.memories['kg'].human_prefix = speaker_name

        return convo

    def list_convo_ids(self, active_only=True) -> List[str]:
        ''' List active convo_ids, from newest to oldest. If active_only = False, include expired convos. '''
        ret = []
        if active_only:
            active = "(@expired:{False})"
        else:
            active = "*"
        # TODO: limit by service + channel
        query = Query(active).dialect(2).return_fields("id")

        for doc in self.redis.ft(self.convo_prefix).search(query).docs:
            ret.append(doc.id.split(':')[3])

        return sorted(ret)

    def get_last_convo_id(self, service, channel) -> str:
        ''' Returns the most recent convo id for this service + channel from Redis '''
        for convo in sorted(self.redis.keys(f'{self.convo_prefix}:*:meta'), reverse=True):
            convo_id = convo.decode().split(':')[3]
            if self.fetch_convo_meta(convo_id, 'service') == service and self.fetch_convo_meta(convo_id, 'channel') == channel:
                return convo_id
        log.debug('No last_convo_id for:', f"{service}|{channel}")
        return None

    def get_last_message_id(self, convo_id) -> str:
        ''' Returns the most recent message for this convo_id from Redis '''
        message_ids = sorted(self.redis.keys(f"{self.convo_prefix}:{convo_id}:lines:*"), reverse=True)
        if message_ids:
            return message_ids[0].decode().split(':')[-1]
        # No messages yet, just return the convo_id
        return convo_id

    def load_convo(self, service, channel, convo_id=None) -> Union[Convo, None]:
        '''
        Load a Convo from Redis.
        If convo_id is None, load the most recent convo from service + channel (if any).
        If no convo is found, return None.
        '''
        if convo_id is None and any([service, channel]) is None:
            raise RuntimeError("load_convo(): specify a convo_id or service + channel")

        if convo_id is None:
            convo_id = self.get_last_convo_id(service, channel)

        if convo_id is None:
            return None

        speaker_name = self.fetch_convo_meta(convo_id, "initiator")

        convo = self.new_convo(service, channel, speaker_name, convo_id)
        convo.memories['summary'].moving_summary_buffer = self.fetch_summary(convo_id)

        self.convos[str(convo)] = convo

        return convo

    def current_convo_id(self, service, channel) -> Union[str, None]:
        ''' Return the current convo_id for service and channel (if any) '''
        if not self.convos:
            # No convos? Load from Redis.
            if not self.load_convo(service, channel):
                log.warning("🤷 No previous convo for:", f"{service}|{channel}")
                return None

        convos = sorted([k for k in self.convos if k.startswith(f'{service}|{channel}|')], reverse=True)
        if not convos:
            return None

        for convo in convos:
            convo_id = convo.split('|', maxsplit=2)[-1]
            if self.convo_expired(convo_id=convo_id):
                log.warning("⏲️  Convo expired:", convo_id)
                self.convos.pop(convo)
                convos.pop(0)

        if not convos:
            return None

        return convo_id

    def get_convo(self, service, channel, convo_id=None) -> Union[Convo, None]:
        '''
        Get a Convo by convo_id or the most recent from service + channel.
        If none exists, return None.
        '''
        if convo_id is None:
            convo_id = self.current_convo_id(service, channel)

        if convo_id is None:
            return None

        convo_key = f"{service}|{channel}|{convo_id}"

        log.debug(self.convos)

        if convo_key not in self.convos:
            self.load_convo(service, channel, convo_id)
        return self.convos[convo_key]

    def expired(self, the_id=None):
        ''' True if time elapsed since the given ulid > conversation_interval, else False '''
        return elapsed(self.id_to_timestamp(the_id), get_cur_ts()) > self.conversation_interval

    @staticmethod
    def id_to_epoch(the_id) -> float:
        ''' Extract the epoch seconds from a ULID '''
        if the_id is None:
            return 0

        if isinstance(the_id, str):
            return ulid.ULID().from_str(the_id).timestamp

        return the_id.timestamp

    def id_to_timestamp(self, the_id):
        ''' Extract the timestamp from a ULID '''
        return get_cur_ts(self.id_to_epoch(the_id))

    def convo_expired(self, service=None, channel=None, convo_id=None) -> bool:
        '''
        True if the timestamp of the last message for this convo is expired, else False.
        '''
        if convo_id is None and not all([service, channel]):
            raise RuntimeError("You must specify a convo_id or both service and channel.")

        if convo_id is None:
            convo_id = self.get_last_convo_id(service, channel)

        # Cache whether the convo is expired
        if self.fetch_convo_meta(convo_id, "expired") == "True":
            return True

        last_msg_id = self.get_last_message_id(convo_id)
        if last_msg_id is None or self.expired(last_msg_id):
            self.set_convo_meta(convo_id, "expired", "True")
            return True

        return False
