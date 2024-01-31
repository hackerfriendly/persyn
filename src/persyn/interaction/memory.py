''' memory.py: long and short term memory by Redis. '''
# pylint: disable=invalid-name, no-name-in-module, abstract-method
import re
import uuid
import datetime as dt

from dataclasses import dataclass
from typing import Union, List, Dict, Any, Optional

import ulid

import redis

from redis.commands.search.query import Query

from langchain.memory import (
    CombinedMemory,
    ConversationSummaryBufferMemory,
    ConversationKGMemory
)

from langchain.vectorstores.redis import Redis

from persyn.interaction.chrono import elapsed, get_cur_ts, seconds_ago
from persyn.interaction.completion import LanguageModel
from persyn.utils.color_logging import ColorLog
from persyn.utils.config import PersynConfig

log = ColorLog()

def escape(text: str) -> str:
    ''' \\ escape all non-word characters '''
    return re.sub(r'(\W)', r'\\\1', text)

def scquery(service: Optional[str] = None, channel: Optional[str] = None) -> str:
    ''' return an escaped redis query for service + channel '''
    ret = []
    if service:
        ret.append("(@service:{" + escape(service) + "})")
    if channel:
        ret.append("(@channel:{" + escape(channel) + "})")
    return  ' '.join(ret)

def decode_dict(d: dict[str, Any]) -> dict:
    ''' Decode a dict fetched from Redis '''
    ret = {}
    for (k, v) in d.items():
        try:
            ret[k.decode()] = v.decode()
        except UnicodeDecodeError:
            ret[k.decode()] = v
    return ret

@dataclass
class Convo:
    ''' Container class for conversations. '''
    service: str
    channel: str
    id: Optional[str] = None

    def __post_init__(self) -> None:
        ''' validate id '''
        if self.id is None:
            self.id = str(ulid.ULID())
        else:
            self.id = str(ulid.ULID().from_str(self.id))

        self.memories = {}              # langchain working memories
        self.visited = set([self.id])   # other convo_ids and ideas we have visited

    def __repr__(self) -> str:
        return f"service='{self.service}', channel='{self.channel}', id='{self.id}'"

    def __str__(self) -> str:
        return f"{self.service}|{self.channel}|{self.id}"

@dataclass
class Recall:
    '''
    Total Recall

    Track conversations. If the conversation_interval has expired, start a new one.

    Also includes helpers for accessing the knowledge graph.
    '''
    config: PersynConfig
    conversation_interval: Optional[int] = None

    def __post_init__(self, conversation_interval: Optional[int] = None) -> None:

        self.bot_name = self.config.id.name
        self.bot_id = uuid.UUID(self.config.id.guid)

        self.conversation_interval = conversation_interval or self.config.memory.conversation_interval
        self.max_summary_size = self.config.memory.max_summary_size

        self.redis = redis.from_url(self.config.memory.redis)

        self.lm = LanguageModel(self.config)

        # indices
        pre = f'persyn2:{self.bot_id}'
        self.convo_prefix = f'{pre}:convo'
        self.opinion_prefix = f'{pre}:opinions'
        self.goal_prefix = f'{pre}:goals'
        self.news_prefix = f'{pre}:news'

        # schemas, required for langchain.redis
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

        # Create indices
        for cmd in [
            f"FT.CREATE {self.convo_prefix} on HASH PREFIX 1 {self.convo_prefix}: SCHEMA service TAG channel TAG expired TAG expired_at NUMERIC role TAG speaker_name TAG initiator TAG verb TAG content TEXT convo_id TEXT feels TEXT content_vector VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE",
            f"FT.CREATE {self.opinion_prefix} on HASH PREFIX 1 {self.opinion_prefix}: SCHEMA service TAG channel TAG opinion TEXT topic TEXT convo_id TAG",
            f"FT.CREATE {self.goal_prefix} on HASH PREFIX 1 {self.goal_prefix}: SCHEMA service TAG channel TAG goal_id TEXT content TEXT achieved NUMERIC content_vector VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE",
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

    def set_convo_meta(self, convo_id: str, k: str, v: str) -> None:
        ''' Set metadata for a conversation '''
        self.redis.hset(f"{self.convo_prefix}:{convo_id}:meta", k, v)

    def incr_convo_meta(self, convo_id: str, k: str, v: Optional[int] = 1) -> int:
        ''' Increment a meta key for a conversation '''
        return self.redis.hincrby(f"{self.convo_prefix}:{convo_id}:meta", k, v)

    def fetch_convo_meta(self, convo_id: str, k: Optional[str] = None) -> Any:
        ''' Fetch a metadata key from a conversation in Redis. If k is not provided, return all metadata. '''
        try:
            if k is None:
                return decode_dict(self.redis.hscan(f'{self.convo_prefix}:{convo_id}:meta')[1].items())
            return self.redis.hget(f"{self.convo_prefix}:{convo_id}:meta", k).decode()
        except AttributeError:
            return None

    def fetch_summary(self, convo_id: str, final=False) -> str:
        '''
        Fetch a conversation summary.
        If final = False, return the most recent conversational memory summary buffer.
        If final = True, return the final (much shorter) summary.
        '''
        if final:
            key="final"
        else:
            key="content"

        ret = self.redis.hget(f"{self.convo_prefix}:{convo_id}:summary", key)
        if ret:
            return ret.decode()
        log.debug("👎 No summary found for:", convo_id)
        return ""

    def new_convo(self, service: str, channel: str, speaker_name: str, convo_id: Optional[str] = None) -> Convo:
        '''
        Start or continue a conversation. If convo_id is not supplied, generate a new one.
        '''
        convo = Convo(
            service=service,
            channel=channel,
            id=convo_id
        )

        if convo_id is None:
            log.warning("🤙 Starting new convo:", convo)

            self.set_convo_meta(convo.id, "service", service)
            self.set_convo_meta(convo.id, "channel", channel)
            self.set_convo_meta(convo.id, "initiator", speaker_name)
            self.set_convo_meta(convo.id, "expired", "False")
            self.set_convo_meta(convo.id, "convo_id", convo.id)

        else:
            speaker_name = self.fetch_convo_meta(convo_id, "initiator") or speaker_name

            if self.convo_expired(convo_id=convo.id):
                log.warning("⚠️  Expired convo:", convo)
            else:
                log.warning("👉 Continuing convo:", convo)

        convo.memories=self.create_lc_memories()
        convo.memories['summary'].human_prefix = speaker_name
        convo.memories['kg'].human_prefix = speaker_name

        return convo

    def list_convo_ids(
        self,
        service: Optional[str] = None,
        channel: Optional[str] = None,
        expired: Optional[bool|None] = None,
        after: Optional[int|None] = None,
        size: Optional[int] = 10
        ) -> Dict[str, Any]:
        '''
        List active convo_ids, from oldest to newest. Constrain to service + channel if provided.
        If expired is True or False, include (or exclude) expired convos. If expired is None, include all convos.
        If after is provided, include only convos that have expired after the given number of seconds.
        Set size to limit the number of results.
        '''
        if expired is None:
            active = " (@expired:{True|False})"
        else:
            active = " (@expired:{" + str(expired) + "})"

        if after:
            since = f" (@expired_at:[{seconds_ago(after)} +inf])"
        else:
            since = " "

        query = Query(
                scquery(service, channel) + active + since
            ).sort_by('convo_id', asc=False).paging(0, size).dialect(2).return_fields('service', 'channel', 'id')

        convos = {}
        docs = self.redis.ft(self.convo_prefix).search(query).docs
        for doc in docs:
            convos[doc.id.split(':')[3]] = {'service': doc.service, 'channel': doc.channel}

        return dict(sorted(convos.items())) # type: ignore

    def get_last_convo_id(self, service: str, channel: str) -> Union[str, None]:
        ''' Returns the most recent convo id for this service + channel from Redis '''

        query = Query(scquery(service, channel)).sort_by('convo_id', asc=False).paging(0, 1).dialect(2).return_fields("id")

        docs = self.redis.ft(self.convo_prefix).search(query).docs

        if not docs:
            log.warning('No last_convo_id for:', f"{service}|{channel}")
            return None

        return docs[0].id.split(':')[3]

    def get_last_message_id(self, convo_id: str) -> str:
        '''
        Returns the most recent message for this convo_id from Redis
        If there are no messages, return the convo_id
        '''
        message_ids = sorted(self.redis.keys(f"{self.convo_prefix}:{convo_id}:dialog:*"), reverse=True)
        if message_ids:
            return message_ids[0].decode().split(':')[-1]
        # No messages yet, just return the convo_id
        return convo_id

    def fetch_convo(self, service: str, channel: str, convo_id: Optional[str] = None) -> Union[Convo, None]:
        '''
        Fetch a Convo from Redis.
        If convo_id is None, load the most recent convo from service + channel (if any).
        If no convo is found, return None.
        '''
        if convo_id is None and any([service, channel]) is None:
            raise RuntimeError("fetch_convo(): specify a convo_id or service + channel")

        if convo_id is None:
            convo_id = self.get_last_convo_id(service, channel)

        if convo_id is None:
            return None

        speaker_name = self.fetch_convo_meta(convo_id, "initiator") or "unknown"
        convo = self.new_convo(service, channel, speaker_name, convo_id)
        summary = self.fetch_summary(convo_id)
        if summary:
            convo.memories['summary'].moving_summary_buffer = summary

        return convo

    def fetch_dialog(self, service: str, channel: str, convo_id: Optional[str] = None) -> str:
        ''' Fetch the convo and return only the current dialog. '''
        convo = self.fetch_convo(service, channel, convo_id)
        if convo is None:
            return ""
        return self.convo_dialog(convo)

    def convo_dialog(self, convo: Convo) -> str:
        ''' Return the current dialog from a Convo object. '''
        return convo.memories['summary'].load_memory_variables({})['history'].replace("System:", "", -1)

    def current_convo_id(self, service: str, channel: str) -> Union[str, None]:
        ''' Return the current convo_id for service and channel (if any) '''
        convo = self.fetch_convo(service, channel)
        if convo is None:
            log.warning("🤷 No previous convo for:", f"{service}|{channel}")
            return None

        if self.fetch_convo_meta(convo.id, "expired") == 'True':
            return None

        return convo.id

    def expired(self, the_id: Optional[str] = None) -> bool:
        ''' True if time elapsed since the given ulid > conversation_interval, else False '''
        return elapsed(self.id_to_timestamp(the_id), get_cur_ts()) > self.conversation_interval # type: ignore

    @staticmethod
    def id_to_epoch(the_id: str) -> float:
        ''' Extract the epoch seconds from a ULID '''
        if the_id is None:
            return 0

        if isinstance(the_id, str):
            return ulid.ULID().from_str(the_id).timestamp

        return the_id.timestamp

    @staticmethod
    def epoch_to_id(epoch: Optional[float] = None) -> str:
        ''' Create a ULID from epoch seconds '''
        if epoch is None:
            epoch = dt.datetime.now(dt.timezone.utc).timestamp()

        return str(ulid.ULID().from_timestamp(epoch))

    def id_to_timestamp(self, the_id: str) -> str:
        ''' Extract the timestamp from a ULID '''
        return get_cur_ts(self.id_to_epoch(the_id))

    def expire_convo(self, convo_id: str) -> None:
        ''' Expire a conversation '''
        self.set_convo_meta(convo_id, "expired", "True")
        self.set_convo_meta(convo_id, "expired_at", dt.datetime.now(dt.timezone.utc).timestamp()) # type: ignore

    def convo_expired(self, service: Optional[str] = None, channel: Optional[str] = None, convo_id: Optional[str] = None) -> bool:
        '''
        True if the convo metadata is expired, or if the timestamp of the last message for this convo is expired. Otherwise False.
        '''
        if convo_id is None and not all([service, channel]):
            raise RuntimeError("You must specify a convo_id or both service and channel.")

        if convo_id is None:
            convo_id = self.get_last_convo_id(service, channel)

        if convo_id is None:
            return True

        if self.fetch_convo_meta(convo_id, "expired") == "True":
            return True

        last_msg_id = self.get_last_message_id(convo_id)
        if last_msg_id is None or self.expired(last_msg_id):
            self.expire_convo(convo_id)
            return True

        return False

    def find_related_convos(
        self,
        service: str,
        channel: str,
        text: str,
        exclude_convo_ids: Optional[List[str]] = None,
        threshold: Optional[float] = 1.0,
        size: Optional[int] = 1
        ) -> List[tuple[str, float]]:
        '''
        Find conversations related to text using vector similarity
        '''
        log.debug(f"find_related_convos: {service} {channel} '{text}' {exclude_convo_ids} {threshold} {size}")

        if not text or len(text) < 4:
            return []

        # TODO: truncate or paginate at 8191 tokens HERE.
        emb = self.lm.get_embedding(text)

        service_channel = "((@service:{$service}) (@channel:{$channel}) (@verb:{summary}))"

        if exclude_convo_ids:
            exclude =  ''.join([f" -(@convo_id:{escape(convo_id)})" for convo_id in exclude_convo_ids])
            query = (
                Query(
                    "(" + service_channel + exclude + ") @content_vector:[VECTOR_RANGE $threshold $emb]=>{$YIELD_DISTANCE_AS: score}"
                )
                .sort_by("score")
                .return_fields("service", "channel", "content", "score")
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "emb": emb, "threshold": threshold}
        else:
            query = (
                Query(
                    service_channel + " @content_vector:[VECTOR_RANGE $threshold $emb]=>{$YIELD_DISTANCE_AS: score}"
                )
                .sort_by("score")
                .return_fields("service", "channel", "convo_id", "content", "score")
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "emb": emb, "threshold": threshold}

        reply = self.redis.ft(self.convo_prefix).search(query, query_params)

        best = ""
        if reply.docs:
            best = f" (best: {float(reply.docs[0].score):0.3f})"

        log.info("👨‍👩‍👧‍👦 find_related_convos():", f"{reply.total} matches, {len(reply.docs)} <= {threshold:0.3f}{best}")

        ret = [(doc.id.split(':')[3], float(doc.score)) for doc in reply.docs]
        log.warning('find_related_convos():', ret)
        return ret
