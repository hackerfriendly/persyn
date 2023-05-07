''' memory.py: long and short term memory by Elasticsearch. '''
# pylint: disable=invalid-name, no-name-in-module, abstract-method, no-member
import uuid

import ulid

import redis
from redis.commands.search.query import Query

from dotwiz import DotWiz  # pylint: disable=no-member

# Time
from persyn.interaction.chrono import elapsed, get_cur_ts

# Embeddings
from persyn.interaction.completion import LanguageModel

# Relationship graphs
from persyn.interaction.relationships import Relationships

# Color logging
from persyn.utils.color_logging import ColorLog

log = ColorLog()


class Recall():
    ''' Total Recall: stm + ltm. '''
    def __init__(self, persyn_config, conversation_interval=None):
        self.bot_name = persyn_config.id.name
        self.bot_id = uuid.UUID(persyn_config.id.guid)

        self.stm = ShortTermMemory(persyn_config, conversation_interval)
        self.ltm = LongTermMemory(persyn_config)

    def save(self, service, channel, msg, speaker_name, speaker_id, verb=None, convo_id=None):
        '''
        Save to stm and ltm. Clears stm if it expired. Returns the current convo_id.

        Specify a different convo_id to override the value in stm.
        '''
        if self.stm.expired(service, channel):
            self.stm.clear(service, channel)

        if convo_id is None:
            convo_id = self.stm.convo_id(service, channel)

        if verb is None:
            verb = 'dialog'

        self.stm.append(
            service,
            channel,
            self.ltm.save_convo(
                service=service,
                channel=channel,
                convo_id=convo_id,
                msg=msg,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                verb=verb
            )
        )
        return convo_id

    def summary(self, service, channel, summary, keywords=None):
        ''' Save a summary. Clears stm. '''
        if not summary:
            return False

        if keywords is None:
            keywords = []

        convo_id = self.stm.convo_id(service, channel)
        self.stm.clear(service, channel)
        self.ltm.save_summary(service, channel, convo_id, summary, keywords)
        return True

    def expired(self, service, channel):
        ''' True if this conversation has expired, else False '''
        return self.stm.expired(service, channel)

    def forget(self, service, channel):
        ''' What were we talking about? '''
        self.stm.clear(service, channel)
        return True

    def add_goal(self, service, channel, goal):
        ''' Add a goal to this channel. '''
        return self.ltm.add_goal(service, channel, goal)

    def achieve_goal(self, service, channel, goal):
        ''' Achieve a goal from this channel. '''
        return self.ltm.achieve_goal(service, channel, goal)

    def get_goals(self, service, channel, goal=None, achieved=None, size=10):
        ''' Return the goals for this channel, if any. '''
        return self.ltm.get_goals(service, channel, goal, achieved, size)

    def list_goals(self, service, channel, achieved=False, size=10):
        ''' Return a simple list of goals for this channel, if any. '''
        return self.ltm.list_goals(service, channel, achieved, size)

    def lookup_summaries(self, service, channel, search, size=1):
        ''' Oh right. '''
        return self.ltm.lookup_summaries(service, channel, search, size)

    def judge(self, service, channel, topic, opinion, convo_id):
        ''' Judge not, lest ye be judged '''
        log.warning(f"👨‍⚖️ judging {topic}")
        return self.ltm.save_opinion(service, channel, topic, opinion, convo_id)

    def surmise(self, service, channel, topic, size=10):
        ''' Everyone's got an opinion '''
        log.warning(f"📌 opinion on {topic}")
        return self.ltm.lookup_opinions(service, channel, topic, size)

    def dialog(self, service, channel):
        ''' Return the dialog from stm (if any) '''
        convo = self.stm.fetch(service, channel)
        if convo:
            return [
                f"{line.speaker_name}: {line.msg}" for line in convo
                if 'verb' not in line or line.verb == 'dialog'
            ]
        return []

    def convo(self, service, channel, feels=False):
        '''
        Return the entire convo from stm (if any).

        Result is a list of "speaker: msg" or "speaker (verb): msg" strings.
        '''
        convo = self.stm.fetch(service, channel)
        if not convo:
            return []

        ret = []
        for msg in convo:
            if msg.verb in ['dialog', None]:
                ret.append(f"{msg.speaker_name}: {msg.msg}")
            elif feels or msg.verb != 'feels':
                ret.append(f"{msg.speaker_name} {msg.verb}: {msg.msg}")

        return ret

    def feels(self, convo_id):
        '''
        Return the last known feels for this channel.
        '''
        convo = self.ltm.get_convo_by_id(convo_id)
        log.debug("📢", convo)
        for doc in convo[::-1]:
            log.debug("📢", doc)
            if doc.verb == 'feels':
                return doc.msg

        return "nothing in particular"

    def summaries(self, service, channel, size=3):
        ''' Return the summary text from ltm (if any) '''
        return [s.summary for s in self.ltm.lookup_summaries(service, channel, None, size=size)]

    def lts(self, service, channel):
        ''' Return the timestamp of the last message from this channel (if any) '''
        return self.ltm.get_last_timestamp(service, channel)

class ShortTermMemory():
    ''' Wrapper class for in-process short term conversational memory. '''
    def __init__(self, persyn_config, conversation_interval=None):
        self.convo = {}
        self.persyn_config = persyn_config
        self.conversation_interval = conversation_interval or persyn_config.memory.conversation_interval

    def _new(self, service, channel):
        ''' Immediately initialize a new channel without sanity checking '''
        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'] = []
        self.convo[service][channel]['id'] = str(ulid.ULID())
        self.convo[service][channel]['opinions'] = []
        log.warning("⚠️  New convo:", self.convo[service][channel]['id'])
        return self.convo[service][channel]

    def exists(self, service, channel):
        ''' True if a channel already exists, else False '''
        return service in self.convo and channel in self.convo[service]

    def create(self, service, channel):
        ''' Create a new empty channel. Erases existing channel if any. '''
        if not self.exists(service, channel):
            if service not in self.convo:
                self.convo[service] = {}
            if channel not in self.convo[service]:
                self.convo[service][channel] = {}
        return self._new(service, channel)

    def clear(self, service, channel):
        ''' Clear a channel. '''
        if not self.exists(service, channel):
            return
        self._new(service, channel)

    def expired(self, service, channel):
        ''' True if time elapsed since last update is > conversation_interval, else False '''
        if not self.exists(service, channel):
            return True
        return elapsed(self.convo[service][channel]['ts'], get_cur_ts()) > self.conversation_interval

    def append(self, service, channel, convo_obj):
        '''
        Append a convo object to a channel. If the current channel expired, clear it first.
        Returns the convo_id.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)
        if self.expired(service, channel):
            self.clear(service, channel)

        self.convo[service][channel]['ts'] = get_cur_ts()
        self.convo[service][channel]['convo'].append(convo_obj)

        return self.convo_id(service, channel)

    def add_bias(self, service, channel, opinion):
        '''
        Append a short-term opinion to a channel if it's not already there. Returns the current opinions.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)
        if opinion not in self.convo[service][channel]['opinions']:
            self.convo[service][channel]['opinions'].append(opinion)
        return self.convo[service][channel]['opinions']

    def get_bias(self, service, channel):
        '''
        Return all short-term opinions for a channel.
        '''
        if not self.exists(service, channel):
            self.create(service, channel)
        return self.convo[service][channel]['opinions']

    def fetch(self, service, channel):
        ''' Fetch the current convo, if any '''
        if not self.exists(service, channel):
            return []
        return self.convo[service][channel]['convo']

    def opinions(self, service, channel):
        ''' Fetch the current opinions expressed in convo, if any '''
        if not self.exists(service, channel):
            return []
        return self.convo[service][channel]['opinions']

    def convo_id(self, service, channel):
        ''' Return the current convo id. Make a new convo if needed. '''
        if not self.exists(service, channel):
            return self.create(service, channel)['id']
        return self.convo[service][channel]['id']

    def last(self, service, channel):
        ''' Fetch the last message from this convo (if any) '''
        if not self.exists(service, channel):
            return None

        last = self.fetch(service, channel)
        if last:
            return last[-1]

        return None

# LTM object
class LongTermMemory(): # pylint: disable=too-many-arguments
    ''' Wrapper class for Elasticsearch conversational memory. '''
    def __init__(self, persyn_config):
        self.relationships = Relationships(persyn_config)
        self.completion = LanguageModel(persyn_config)
        self.persyn_config = persyn_config
        self.bot_name = persyn_config.id.name
        self.bot_id = uuid.UUID(persyn_config.id.guid)
        self.bot_ulid = ulid.ULID().from_uuid(self.bot_id)

        self.redis = redis.from_url(persyn_config.memory.redis)

        self.convo_prefix = f'persyn:{self.bot_id}:convo'
        self.summary_prefix = f'persyn:{self.bot_id}:summary'
        self.opinion_prefix = f'persyn:{self.bot_id}:opinion'

        for cmd in [
            f"FT.CREATE {self.convo_prefix} on HASH PREFIX 1 {self.convo_prefix}: SCHEMA service TAG channel TAG convo_id TAG speaker_name TEXT speaker_id TEXT msg TEXT verb TAG emb VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE",
            f"FT.CREATE {self.summary_prefix} on HASH PREFIX 1 {self.summary_prefix}: SCHEMA service TAG channel TAG convo_id TAG summary TEXT keywords TAG emb VECTOR HNSW 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE",
            f"FT.CREATE {self.opinion_prefix} on HASH PREFIX 1 {self.opinion_prefix}: SCHEMA service TAG channel TAG opinion TEXT topic TAG convo_id TAG",
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

    @staticmethod
    def entity_id_to_timestamp(entity_id):
        ''' Extract the timestamp from a ULID '''
        if entity_id is None:
            return entity_id

        if isinstance(entity_id, str):
            return ulid.ULID().from_str(entity_id).timestamp

        return entity_id.timestamp

    def shortest_path(self, src, dest, src_type=None, dest_type=None):
        ''' TODO: REMOVE THIS STUB AND REPLACE WITH graph.py '''
        return []

    def get_last_timestamp(self, service, channel):
        '''
        Get the timestamp of the last message, or the current ts if there is none.
        '''
        msg = self.get_last_message(service, channel)
        if msg:
            return get_cur_ts(epoch=ulid.ULID().from_str(msg.pk).timestamp)

        return get_cur_ts()

    def save_convo(
        self,
        service,
        channel,
        speaker_name,
        msg,
        convo_id=None,
        speaker_id=None,
        verb='dialog'
    ):
        '''
        Save a line of conversation. Returns the Convo object.
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

    def save_summary(self, service, channel, convo_id, summary, keywords=None):
        '''
        Save a conversation summary to memory. Returns the Summary object.
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

        return DotWiz(ret)


    def get_last_message(self, service, channel):
        ''' Return the last message seen on this channel '''
        query = (
            Query(
                """(@service:{$service}) (@channel:{$channel}) (@verb:{dialog})"""
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
        try:

            return self.redis.ft(self.convo_prefix).search(query, query_params).docs[0]
        except IndexError:
            return None

    def get_convo_by_id(self, convo_id):
        ''' Return all Convo objects matching convo_id in chronological order '''
        query = Query("(@convo_id:{$convo_id})").dialect(2)
        query_params = {"convo_id": convo_id}
        return self.redis.ft(self.convo_prefix).search(query, query_params).docs

    def get_summary_by_id(self, convo_id):
        ''' Return the last summary for this convo_id '''
        query = Query("(@convo_id:{$convo_id})").dialect(2)
        query_params = {"convo_id": convo_id}
        try:
            return self.redis.ft(self.summary_prefix).search(query, query_params).docs[-1]
        except IndexError:
            return None

    def lookup_summaries(self, service, channel, search=None, size=3):
        '''
        Return a list of summaries matching the search term for this channel.
        '''
        log.warning(f"lookup_summaries(): {service} {channel} {search} {size}")

        if search is None:
            query = Query("(@service:{$service}) (@channel:{$channel})").paging(0, size).dialect(2)
            query_params = {"service": service, "channel": channel}
            return self.redis.ft(self.summary_prefix).search(query, query_params).docs

        # summary is a text field, so tokenization and stemming apply.
        query = Query('(@service:{$service}) (@channel:{$channel}) (@summary:$summary)').paging(0, size).dialect(2)
        query_params = {"service": service, "channel": channel, "summary": search}
        return self.redis.ft(self.summary_prefix).search(query, query_params).docs

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

        log.info(f"💾 Opinion of {topic}:", opinion)

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

        log.debug(f"💾 Opinion of {topic}:", opinion)

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

    def find_related_convos(self, service, channel, convo, current_convo_id=None, size=1, threshold=1.0):
        '''
        Find conversations related to convo using vector similarity
        '''
        emb = self.completion.model.get_embedding(' '.join(convo))

        if current_convo_id is None:
            query = (
                Query(
                    "((@service:{$service}) (@channel:{$channel}) (@verb:{dialog}))=>[KNN " + str(size) + " @emb $emb as score]"
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
                    "((@service:{$service}) (@channel:{$channel}) (@verb:{dialog}) -(@convo_id:{$convo_id}))=>[KNN " + str(size) + " @emb $emb as score]"
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
                ret.append(doc)

        log.info("👨‍👩‍👧‍👦 find_related_convos():", f"{reply.total} matches, {len(ret)} < {threshold}")
        return ret

    def add_goal(self, service, channel, goal, refresh=True):
        '''
        Add a goal to a channel. Returns the top 10 unachieved goals.
        '''
        return []
        # cur_ts = get_cur_ts()
        # doc = {
        #     "service": service,
        #     "channel": channel,
        #     "@timestamp": cur_ts,
        #     "goal": goal,
        #     "achieved": False
        # }
        # _id = self.es.index(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        #     index=self.index['goal'],
        #     document=doc,
        #     refresh='true' if refresh else 'false'
        # )["_id"]

        # log.debug("doc:", _id)

        # return self.get_goals(service, channel, achieved=False)

    def achieve_goal(self, service, channel, goal):
        '''
        Set a goal to the achieved state. Returns the top ten unachieved goals.
        '''
        return []
        # for doc in self.get_goals(service, channel, goal, achieved=False):
        #     doc['_source']['achieved'] = True
        #     doc['_source']['achieved_on'] = get_cur_ts()
        #     self.es.update(index=self.index['goal'], id=doc['_id'], doc=doc['_source'], refresh=True)

        # return self.get_goals(service, channel, achieved=False)

    def get_goals(self, service, channel, goal=None, achieved=None, size=10):
        '''
        Return goals for a channel. Returns the 10 most recent goals by default.
        Set achieved to True or False to return only achieved or unachieved goals.
        Specify a goal to return only that specific goal.
        '''
        ret = []
        return ret

        # query = {
        #     "bool": {
        #         "must": [
        #             {"match": {"service.keyword": service}},
        #             {"match": {"channel.keyword": channel}},
        #         ]
        #     }
        # }
        # if goal:
        #     query["bool"]["must"].append({"match": {"goal": goal}})
        # if achieved is not None:
        #     query["bool"]["must"].append({"match": {"achieved": achieved}})

        # ret = self.es.search(  # pylint: disable=unexpected-keyword-arg
        #     index=self.index['goal'],
        #     query=query,
        #     sort=[{"@timestamp": {"order": "desc"}}],
        #     size=size
        # )['hits']['hits']

        # log.debug(f"🥇 return: {ret}")
        # return ret

    def list_goals(self, service, channel, achieved=False, size=10):
        '''
        Return a simple list of goals for a channel. Returns the 10 most recent goals by default.
        '''
        ret = []
        for goal in self.get_goals(service, channel, goal=None, achieved=achieved, size=size):
            ret.append(goal.goal)
        return ret

    def add_news(self, service, channel, url, title, refresh=True):
        '''
        Add a news url that we've read. Returns the doc _id.
        '''
        return None
        # cur_ts = get_cur_ts()
        # doc = {
        #     "service": service,
        #     "channel": channel,
        #     "@timestamp": cur_ts,
        #     "url": url,
        #     "title": title
        # }
        # _id = self.es.index(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        #     index=self.index['news'],
        #     document=doc,
        #     refresh='true' if refresh else 'false'
        # )["_id"]

        # log.debug("🗞️ doc:", _id)

        # return _id

    def have_read(self, service, channel, url):
        '''
        Return True if we have read this article, otherwise False.
        '''
        ret = []
        return False

        # query = {
        #     "bool": {
        #         "must": [
        #             {"match": {"service.keyword": service}},
        #             {"match": {"channel.keyword": channel}},
        #             {"match": {"url.keyword": url}}
        #         ]
        #     }
        # }

        # ret = bool(self.es.search(  # pylint: disable=unexpected-keyword-arg
        #     index=self.index['news'],
        #     query=query,
        #     sort=[{"@timestamp": {"order": "desc"}}],
        #     size=1
        # )['hits']['hits'])

        # log.debug(f"🗞️ {url}: {ret}")
        # return ret
