''' memory.py: long and short term memory by Redis. '''
# pylint: disable=invalid-name, no-name-in-module, abstract-method, no-member
import uuid

import re
import ulid

import redis
from redis.commands.search.query import Query

from dotwiz import DotWiz  # pylint: disable=no-member

from neomodel import DateTimeProperty, StringProperty, UniqueIdProperty, FloatProperty, IntegerProperty, RelationshipTo, StructuredRel, Q
from neomodel import config as neomodel_config
from neomodel import db as neomodel_db
from neomodel.contrib import SemiStructuredNode

# Time
from persyn.interaction.chrono import elapsed, get_cur_ts

# Embeddings
from persyn.interaction.completion import LanguageModel

# Relationship graphs
from persyn.interaction.relationships import Relationships

# Color logging
from persyn.utils.color_logging import ColorLog

log = ColorLog()

# Neomodel (Neo4j) graph classes
class Entity(SemiStructuredNode):
    ''' Superclass for everything '''
    name = StringProperty(required=True)
    bot_id = StringProperty(required=True)

class PredicateRel(StructuredRel):
    ''' Predicate relationship: s P o '''
    weight = IntegerProperty(default=0)
    verb = StringProperty(required=True)

class Person(Entity):
    ''' Something you can have a conversation with '''
    created = DateTimeProperty(default_now=True)
    last_contact = DateTimeProperty(default_now=True)
    entity_id = UniqueIdProperty()
    speaker_id = UniqueIdProperty()
    link = RelationshipTo('Entity', 'LINK', model=PredicateRel)

class Thing(Entity):
    ''' Any old thing '''
    created = DateTimeProperty(default_now=True)
    link = RelationshipTo('Entity', 'LINK', model=PredicateRel)

class Human(Person):
    ''' Flesh and blood '''
    last_contact = DateTimeProperty(default_now=True)
    trust = FloatProperty(default=0.0)

class Bot(Person):
    ''' Silicon and electricity '''
    service = StringProperty(required=True)
    channel = StringProperty(required=True)
    trust = FloatProperty(default=0.0)


class Recall():
    '''
    Total Recall

    Track conversations. If the conversation_interval has expired, start a new one.

    Also builds the knowledge graph.
    '''
    def __init__(self, persyn_config, conversation_interval=None):

        self.persyn_config = persyn_config

        self.bot_name = persyn_config.id.name
        self.bot_id = uuid.UUID(persyn_config.id.guid)
        self.bot_ulid = ulid.ULID().from_uuid(self.bot_id)

        self.conversation_interval = conversation_interval or persyn_config.memory.conversation_interval

        self.relationships = Relationships(persyn_config)
        self.completion = LanguageModel(persyn_config)

        self.redis = redis.from_url(persyn_config.memory.redis)

        # indices
        self.convo_prefix = f'persyn:{self.bot_id}:convo'
        self.summary_prefix = f'persyn:{self.bot_id}:summary'
        self.opinion_prefix = f'persyn:{self.bot_id}:opinion'
        self.goal_prefix = f'persyn:{self.bot_id}:goal'
        self.news_prefix = f'persyn:{self.bot_id}:news'

        # sets
        self.active_convos_prefix = f"{self.convo_prefix}:active_convos"

        # clear any active convos
        self.redis.delete(self.active_convos_prefix)

        # Create indices
        for cmd in [
            f"FT.CREATE {self.convo_prefix} on HASH PREFIX 1 {self.convo_prefix}: SCHEMA service TAG channel TAG convo_id TAG speaker_name TEXT speaker_id TEXT msg TEXT verb TAG emb VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE",
            f"FT.CREATE {self.summary_prefix} on HASH PREFIX 1 {self.summary_prefix}: SCHEMA service TAG channel TAG convo_id TAG summary TEXT keywords TAG emb VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE",
            f"FT.CREATE {self.opinion_prefix} on HASH PREFIX 1 {self.opinion_prefix}: SCHEMA service TAG channel TAG opinion TEXT topic TAG convo_id TAG",
            f"FT.CREATE {self.goal_prefix} on HASH PREFIX 1 {self.goal_prefix}: SCHEMA service TAG channel TAG goal TAG",
            f"FT.CREATE {self.news_prefix} on HASH PREFIX 1 {self.news_prefix}: SCHEMA service TAG channel TAG url TAG title TEXT",
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

        # Convenience ids. Useful when browsing with RedisInsight.
        self.redis.hset(f"persyn:{self.bot_id}:whoami", "bot_name", self.bot_name)
        self.redis.hset(f"persyn:{self.bot_id}:whoami", "bot_id", str(self.bot_id))
        self.redis.hset(f"persyn:{self.bot_id}:whoami", "bot_ulid", str(self.bot_ulid))

        if hasattr(persyn_config.memory, 'neo4j'):
            neomodel_config.DATABASE_URL = persyn_config.memory.neo4j.url


    def list_convos(self):
        ''' Return the set of all active convos for all services + channels '''
        return self.redis.smembers(self.active_convos_prefix)

    def judge(self, service, channel, topic, opinion, convo_id):
        ''' Judge not, lest ye be judged '''
        log.warning(f"👨‍⚖️ judging {topic}")
        return self.save_opinion(service, channel, topic, opinion, convo_id)

    def surmise(self, service, channel, topic, size=10):
        ''' Everyone's got an opinion '''
        log.warning(f"📌 opinion on {topic}")
        return self.lookup_opinions(service, channel, topic, size)

    def convo(self, service, channel, convo_id=None, feels=False, verb=None, raw=False, size=1000):
        '''
        Return an entire convo.

        If convo_id is None, use the current convo (if any).

        If feels is True, also include feelings.

        If verb is not None, filter to only include messages with a matching verb.

        If raw is true, return convo objects. Otherwise return a list of strings.
        '''
        if convo_id is None:
            lm = self.get_last_message(service, channel)
            if not lm:
                return []
            convo_id = lm.convo_id

        if verb:
            query = (
                Query(
                    """(@service:{$service}) (@channel:{$channel}) (@convo_id:{$convo_id}) (@verb:{$verb})"""
                )
                .return_fields(
                    "speaker_name",
                    "msg",
                    "verb",
                    "convo_id"
                )
                .sort_by("convo_id", asc=True)
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "convo_id": convo_id, "verb": verb}
        else:
            query = (
                Query(
                    """(@service:{$service}) (@channel:{$channel}) (@convo_id:{$convo_id})"""
                )
                .return_fields(
                    "speaker_name",
                    "msg",
                    "verb",
                    "convo_id"
                )
                .sort_by("convo_id", asc=True)
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "convo_id": convo_id}

        ret = []
        realization = None
        for msg in self.redis.ft(self.convo_prefix).search(query, query_params).docs:
            if not getattr(msg, 'verb', None):
                continue

            if msg.verb == 'new_convo':
                continue
            if msg.verb == 'realizes':
                realization = f"{msg.speaker_name} {msg.verb}: {msg.msg}"
                continue

            if msg.verb == 'dialog':
                if raw:
                    ret.append(msg)
                else:
                    ret.append(f"{msg.speaker_name}: {msg.msg}")
            elif feels or msg.verb != 'feels':
                if raw:
                    ret.append(msg)
                else:
                    ret.append(f"{msg.speaker_name} {msg.verb}: {msg.msg}")

        if realization:
            ret.append(realization)

        return ret

    def feels(self, convo_id):
        '''
        Return the last known feels for this convo_id.
        '''
        query = (
            Query(
                """(@convo_id:{$convo_id}) (@verb:{feels})"""
            )
            .return_fields(
                "msg"
            )
            .sort_by("convo_id", asc=False)
            .paging(0, 1)
            .dialect(2)
        )
        query_params = {"convo_id": convo_id}

        ret = self.redis.ft(self.convo_prefix).search(query, query_params).docs
        if ret:
            return ret[0].msg

        return "nothing in particular"

    def save_convo_line(
        self,
        service,
        channel,
        msg,
        speaker_name,
        speaker_id=None,
        convo_id=None,
        verb='dialog'
    ):
        '''
        Save a line of conversation.
        If convo_id is None, a new ID is assigned.
        All convo lines have a ULID primary key, embedding the timestamp of each message.
        The redis key is:
          persyn:[bot uuid]:convo:[convo id]:[pk]
        Returns the Convo object.
        '''
        # Save speaker entity
        if speaker_id is None:
            # speaker_id = self.get_speaker_id(service, channel, speaker)
            speaker_id = "unknown speaker"

        if convo_id is None:
            convo_id = str(ulid.ULID())

        pk = str(ulid.ULID())

        ret = {
            "service": service,
            "channel": channel,
            "convo_id": convo_id,
            "speaker_name": speaker_name,
            "speaker_id": speaker_id,
            "msg": msg,
            "verb": verb,
            "pk": pk,
            "emb": self.completion.model.get_embedding(msg)
        }

        key = f"{self.convo_prefix}:{convo_id}:{pk}"
        for k, v in ret.items():
            try:
                self.redis.hset(key, k, v)
            except redis.exceptions.DataError as err:
                log.error(f"{err}:", f"{key} | {k} | {v}")
                raise err

        log.debug(f"💾 Convo line saved for {key}:", ret['pk'])

        return DotWiz(ret)


    def new_convo(self, service, channel):
        ''' Start a new conversation. Returns the convo_id. '''

        ret = self.save_convo_line(
            service,
            channel,
            speaker_name=self.bot_name,
            msg="new_convo",
            convo_id=None,
            speaker_id=str(self.bot_ulid),
            verb="new_convo"
        )

        log.warning("⚠️  New convo:", ret.convo_id)

        # No need to remove the previous conversation. It will be removed when cns summarizes it.
        self.redis.sadd(self.active_convos_prefix, f"{service}|{channel}|{ret.convo_id}")

        return ret.convo_id

    def expired(self, service, channel):
        ''' True if time elapsed since the last convo line is > conversation_interval, else False '''
        return elapsed(self.get_last_timestamp(service, channel), get_cur_ts()) > self.conversation_interval

    def convo_id(self, service, channel):
        ''' Return the current convo id. Make a new convo if needed. '''

        if self.expired(service, channel):
            return self.new_convo(service, channel)

        ret = self.get_last_message(service, channel)
        if ret:
            return ret.convo_id
        return None

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

    def save_summary(self, service, channel, convo_id, summary, keywords=None):
        '''
        Save a conversation summary to memory. This also ends the current convo.
        Returns the Summary object.
        '''
        if keywords is None:
            keywords = []

        ret = {
            "service": service,
            "channel": channel,
            "convo_id": convo_id,
            "summary": summary,
            "keywords": ','.join(keywords),
            "emb": self.completion.model.get_embedding(summary)
        }

        # Don't namespace on pk like we do for convo.
        # This will overwrite the conversation summary each time a new one is generated.
        for k, v in ret.items():
            self.redis.hset(f"{self.summary_prefix}:{convo_id}", k, v)

        self.new_convo(service, channel)

        return DotWiz(ret)


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
                "speaker_id",
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

    def get_convo_by_id(self, convo_id, size=1000):
        ''' Return all Convo objects matching convo_id in chronological order '''
        query = Query("(@convo_id:{$convo_id})").paging(0, size).dialect(2)
        query_params = {"convo_id": convo_id}
        # not sure how to sort by pk, so do it manually
        return sorted(self.redis.ft(self.convo_prefix).search(query, query_params).docs, key=lambda k: k['pk'])

    def get_summary_by_id(self, convo_id):
        ''' Return the last summary for this convo_id '''
        query = Query("(@convo_id:{$convo_id})").sort_by("convo_id", asc=False).paging(0, 1).dialect(2)
        query_params = {"convo_id": convo_id}
        try:
            return self.redis.ft(self.summary_prefix).search(query, query_params).docs[0]
        except IndexError:
            return None

    def summaries(self, service, channel, search=None, size=3, raw=False):
        '''
        Return a list of summaries matching the search term for this channel.

        If raw is False, return a list of strings.
        If raw is True, return the summary objects.
        '''
        log.debug(f"recall.summaries(): {service} {channel} {search} {size}")

        if search is None:
            query = (
                Query(
                    "(@service:{$service}) (@channel:{$channel})"
                )
                .sort_by("convo_id", asc=False)
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel}
            if raw:
                return self.redis.ft(self.summary_prefix).search(query, query_params).docs[::-1]
            return [doc.summary for doc in self.redis.ft(self.summary_prefix).search(query, query_params).docs][::-1]

        # summary is a text field, so tokenization and stemming apply.
        query = (
            Query(
                '(@service:{$service}) (@channel:{$channel}) (@summary:$summary)'
            )
            .sort_by("convo_id", asc=False)
            .paging(0, size)
            .dialect(2)
        )
        query_params = {"service": service, "channel": channel, "summary": search}
        if raw:
            return self.redis.ft(self.summary_prefix).search(query, query_params).docs[::-1]
        return [doc.summary for doc in self.redis.ft(self.summary_prefix).search(query, query_params).docs][::-1]

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

    def save_opinion(self, service, channel, topic, opinion, convo_id):
        '''
        Save an opinion to Redis.
        '''
        if convo_id is None:
            log.warning("save_opinion(): no convo_id, skipping.")
            return

        ret = {
            "service": service,
            "channel": channel,
            "topic": topic,
            "opinion": opinion,
            "convo_id": convo_id
        }

        # One opinion per topic, for each service+channel+convo_id.
        # Opinions accumulate over more conversations.
        topic_id = self.name_to_entity_id(service, channel, topic)
        for k, v in ret.items():
            self.redis.hset(f"{self.opinion_prefix}:{convo_id}:{topic_id}", k, v)

        log.debug(f"📌 Opinion of {topic}:", opinion)

    def lookup_opinions(self, service, channel, topic, size=10):
        ''' Look up an opinion in Redis. '''
        log.debug(f"lookup_opinions(): {service} {channel} {topic} {size}")

        query = (
            Query(
                '(@service:{$service}) (@channel:{$channel}) (@topic:{$topic})'
            )
            .sort_by("convo_id", asc=False)
            .paging(0, size)
            .dialect(2)
        )

        query_params = {"service": service, "channel": channel, "topic": topic}
        return [doc.opinion for doc in self.redis.ft(self.opinion_prefix).search(query, query_params).docs]

    def find_related_convos(self, service, channel, query, current_convo_id=None, size=1, threshold=1.0, any_convo=True):
        '''
        Find conversations related to query using vector similarity
        '''
        # TODO: truncate to 8191 tokens HERE.
        emb = self.completion.model.get_embedding(query)

        if any_convo:
            service_channel = "((@service:{$service}))"
        else:
            service_channel = "((@service:{$service}) (@channel:{$channel}))"

        if current_convo_id is None:
            query = (
                Query(
                    f"{service_channel}=>[KNN " + str(size) + " @emb $emb as score]"
                )
                .sort_by("score")
                .return_fields("service", "channel", "convo_id", "msg", "speaker_name", "pk", "score")
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "emb": emb}

        else:
            # exclude the current convo_id
            query = (
                Query(
                   "(" + service_channel + "-(@convo_id:{$convo_id}))=>[KNN " + str(size) + " @emb $emb as score]"
                )
                .sort_by("score")
                .return_fields("service", "channel", "convo_id", "msg", "speaker_name", "pk", "score")
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "emb": emb, "convo_id": current_convo_id}

        reply = self.redis.ft(self.convo_prefix).search(query, query_params)
        ret = []
        for doc in reply.docs:
            # Redis uses 1-cosine_similarity, so it's a distance (not a similarity)
            if float(doc.score) < threshold:
                log.debug("👨‍👩‍👧 Related: ", doc.msg)
                ret.append(doc)

        best = ""
        if reply.docs:
            best = f" (best: {float(reply.docs[0].score):0.3f})"

        log.info("👨‍👩‍👧‍👦 find_related_convos():", f"{reply.total} matches, {len(ret)} < {threshold:0.3f}{best}")

        return ret

    def find_related_summaries(self, service, channel, query, current_convo_id=None, size=1, threshold=1.0, any_convo=True):
        '''
        Find summaries related to query using vector similarity
        '''
        # TODO: truncate to 8191 tokens HERE.
        emb = self.completion.model.get_embedding(query)

        if any_convo:
            service_channel = "((@service:{$service}))"
        else:
            service_channel = "((@service:{$service}) (@channel:{$channel}))"

        if current_convo_id is None:
            query = (
                Query(
                    f"{service_channel}=>[KNN " + str(size) + " @emb $emb as score]"
                )
                .sort_by("score")
                .return_fields("service", "channel", "convo_id", "summary", "score")
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "emb": emb}

        else:
            # exclude the current convo_id
            query = (
                Query(
                   "(" + service_channel + "-(@convo_id:{$convo_id}))=>[KNN " + str(size) + " @emb $emb as score]"
                )
                .sort_by("score")
                .return_fields("service", "channel", "convo_id", "summary", "score")
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "emb": emb, "convo_id": current_convo_id}

        reply = self.redis.ft(self.summary_prefix).search(query, query_params)
        ret = []
        for doc in reply.docs:
            # Redis uses 1-cosine_similarity, so it's a distance (not a similarity)
            if float(doc.score) < threshold:
                log.debug("👨‍👩‍👧 Related: ", doc.summary)
                ret.append(doc)

        best = ""
        if reply.docs:
            best = f" (best: {float(reply.docs[0].score):0.3f})"

        log.info("👨‍👩‍👧‍👦 find_related_summaries():", f"{reply.total} matches, {len(ret)} < {threshold:0.3f}{best}")

        return ret

    def add_goal(self, service, channel, goal):
        '''
        Add a goal to a channel.
        '''
        ret = {
            "service": service,
            "channel": channel,
            "goal": goal,
        }

        goal_id = self.name_to_entity_id(service, channel, goal)
        for k, v in ret.items():
            self.redis.hset(f"{self.goal_prefix}:{goal_id}", k, v)

        log.info("⚽️ New goal:", goal)

    def achieve_goal(self, service, channel, goal):
        '''
        Achieve a goal (ie. delete it).
        '''
        goal_id = self.name_to_entity_id(service, channel, goal)

        self.redis.delete(f"{self.goal_prefix}:{goal_id}")

        log.info("🥅 Goal achieved:", goal)

    def get_goals(self, service, channel, goal=None, size=10):
        '''
        Return goals for a channel. Returns the 10 most recent goals by default.
        Specify a goal to return only that specific goal.
        '''
        if goal is None:
            query = (
                Query(
                    """(@service:{$service}) (@channel:{$channel})"""
                )
                .return_fields(
                    "service",
                    "channel",
                    "goal",
                )
                .sort_by("goal", asc=False)
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel}
        else:
            query = (
                Query(
                    """(@service:{$service}) (@channel:{$channel}) (@goal:{goal})"""
                )
                .return_fields(
                    "service",
                    "channel",
                    "goal",
                )
                .sort_by("goal", asc=False)
                .paging(0, size)
                .dialect(2)
            )
            query_params = {"service": service, "channel": channel, "goal": goal}

        try:
            return self.redis.ft(self.goal_prefix).search(query, query_params).docs
        except IndexError:
            return None


    def list_goals(self, service, channel, size=10):
        '''
        Return a simple list of goals for a channel. Returns the 10 most recent goals by default.
        '''
        return [goal.goal for goal in self.get_goals(service, channel, goal=None, size=size)]

    def add_news(self, service, channel, url, title):
        '''
        Add a news url that we've read.
        '''
        ret = {
            "service": service,
            "channel": channel,
            "url": url,
            "title": title,
        }

        title_id = self.name_to_entity_id(service, channel, title)
        for k, v in ret.items():
            self.redis.hset(f"{self.news_prefix}:{title_id}", k, v)

        log.debug("📰 Read the news:", title)
        return True

    def have_read(self, service, channel, url):
        '''
        Return True if we have read this article, otherwise False.
        '''
        query = (
            Query(
                '(@service:{$service}) (@channel:{$channel}) (@url:{$url})'
            )
            .paging(0, 1)
            .dialect(2)
        )

        query_params = {"service": service, "channel": channel, "url": url}
        return bool(self.redis.ft(self.news_prefix).search(query, query_params).total)

    ##
    # Graph functions. This should probably be its own module.

    def delete_all_nodes(self, confirm=False):
        ''' Delete all graph nodes for this bot '''
        if confirm is not True:
            return False

        for node in Entity.nodes.filter(Q(bot_id=self.bot_id)):
            node.delete()

        return True

    def fetch_all_nodes(self, node_type=None):
        ''' Return all graph nodes for this bot '''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'person':
            return Person.nodes.filter(Q(bot_id=self.bot_id))
        if node_type == 'thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    def find_node(self, name, node_type=None):
        ''' Return all nodes with the given name'''
        if node_type is None:
            return Entity.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'person':
            return Person.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        if node_type == 'thing':
            return Thing.nodes.filter(Q(bot_id=self.bot_id), Q(name=name))
        raise RuntimeError(f'Invalid node_type: {node_type}')

    @staticmethod
    def safe_name(name):  # TODO: unify this with gpt.py
        ''' Return name sanitized as alphanumeric, space, or comma only, max 64 characters. '''
        return re.sub(r"[^a-zA-Z0-9, ]+", '', name.strip())[:64]

    def shortest_path(self, src, dest, src_type=None, dest_type=None):
        '''
        Find the shortest path between two nodes, if any.
        If src_type or dest_type are specified, constrain the search to nodes of that type.
        Returns a list of triples (string names of nodes and edges) encountered along the path.
        '''
        safe_src = self.safe_name(src)
        safe_dest = self.safe_name(dest)

        if safe_src == safe_dest:
            return []

        query = f"""
        MATCH
        (a{':'+src_type if src_type else ''} {{name: '{safe_src}', bot_id: '{self.bot_id}'}}),
        (b{':'+dest_type if dest_type else ''} {{name: '{safe_dest}', bot_id: '{self.bot_id}'}}),
        p = shortestPath((a)-[*]-(b))
        WITH p
        WHERE length(p) > 1
        RETURN p
        """
        paths = neomodel_db.cypher_query(query)[0]
        ret = []
        if not paths:
            return ret

        for r in paths[0][0].relationships:
            ret.append((r.start_node.get('name'), r.get('verb'), r.end_node.get('name')))

        return ret

    def triples_to_kg(self, triples):
        '''
        Convert subject, predicate, object triples into a Neo4j graph.
        '''
        if not hasattr(self.persyn_config.memory, 'neo4j'):
            log.error('፨ No graph server defined, cannot call triples_to_kg()')
            return

        speaker_names = {p.name for p in Person.nodes.all() if p.bot_id == self.bot_id}
        thing_names = set()
        speakers = {}

        for triple in triples:
            (s, _, o) = triple

            if s not in speaker_names:
                thing_names.add(s)

            if o not in speaker_names:
                thing_names.add(o)

        with neomodel_db.transaction:
            for name in speaker_names:
                try:
                    speakers[name] = Person.nodes.get(name=name, bot_id=self.bot_id)  # pylint: disable=no-member
                # If they don't yet exist in the graph, make a new node
                except Person.DoesNotExist:  # pylint: disable=no-member
                    speakers[name] = Person(name=name, bot_id=self.bot_id).save()

            things = {}
            for t in Thing.get_or_create(*[{'name': n, 'bot_id': self.bot_id} for n in list(thing_names) if n not in speakers]):
                things[t.name] = t

            for link in triples:
                if link[0] in speakers:
                    subj = speakers[link[0]]
                else:
                    subj = things[link[0]]

                pred = link[1]

                if link[2] in speakers:
                    obj = speakers[link[2]]
                else:
                    obj = things[link[2]]

                found = False
                for rel in subj.link.all_relationships(obj):
                    if rel.verb == pred:
                        found = True
                        rel.weight = rel.weight + 1
                        rel.save()

                if not found:
                    rel = subj.link.connect(obj, {'verb': pred})
