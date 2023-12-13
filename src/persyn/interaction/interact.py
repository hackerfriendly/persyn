'''
interact.py

The limbic system library.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order
import random
import re # used by custom filters

from urllib.parse import urlparse
from typing import Optional, List

import requests

from pydantic import Field

# Long and short term memory
from persyn.interaction.memory import Recall

# Time handling
from persyn.interaction.chrono import exact_time, natural_time, ago, today, elapsed, get_cur_ts

# Prompt completion
from persyn.interaction.completion import LanguageModel

# Custom langchain
from persyn.langchain.zim import ZimWrapper

# Color logging
from persyn.utils.color_logging import log

rs = requests.Session()

def doiify(text):
    ''' Turn DOIs into doi.org links '''
    return re.sub(
        r"(https?://doi\.org/)?(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)",
        lambda match: match.group(0) if match.group(2) is None else f'https://doi.org/{match.group(2)}',
        text
    ).rstrip('.')

class Interact():
    '''
    The Interact class contains all language handling routines and maintains
    the short and long-term memories for each service+channel.
    '''
    def __init__(self, persyn_config):
        self.config = persyn_config

        # Pick a language model for completion
        self.completion = LanguageModel(config=persyn_config)

        # Identify a custom reply filter, if any. Should be a Python expression
        # that receives the reply as the variable 'reply'. (Also has access to
        # 'self' technically and anything else loaded at module scope, so be
        # careful.)
        self.custom_filter = None
        if hasattr(self.config.interact, "filter"):
            assert re # prevent "unused import" type linting
            self.custom_filter = eval(f"lambda reply: {self.config.interact.filter}") # pylint: disable=eval-used

        # Then create the Recall object (short-term, long-term, and graph memory).
        self.recall = Recall(persyn_config)

        # Langchain
        self.llm = self.completion.model.completion_llm

        # # zim / kiwix data sources
        # if self.config.get('zim'):
        #     for cfgtool in self.config.zim:
        #         log.info("💿 Loading zim:", cfgtool)
        #         zim = Tool(
        #                 name=str(cfgtool),
        #                 func=ZimWrapper(path=self.config.zim.get(cfgtool).path).run,
        #                 description=self.config.zim.get(cfgtool).description
        #             )
        #         agent_tools.append(zim)
        #         vector_tools.append(zim)

        # Other tools: introspection (assess software + hardware), Claude (ask for facts)

        self.enc = self.completion.model.get_enc()

    def summarize_convo(
        self,
        service,
        channel,
        save=True,
        include_keywords=False,
        context_lines=0,
        dialog_only=True,
        convo_id=None
    ):
        '''
        Generate a summary of the current conversation for this channel.
        Also generate and save opinions about detected topics.

        If save == True, save convo to long term memory and generate
        knowledge graph nodes (via the autobus).

        Returns the text summary.
        '''
        if convo_id is None:
            convo_id = self.recall.convo_id(service, channel)
            log.warning(f"∑ summarize_convo: {convo_id}")
        if not convo_id:
            log.error("∑ summarize_convo: no convo_id")
            return ""

        log.warning(f"{service} | {channel} | {convo_id}")
        if dialog_only:
            text = self.recall.convo(service, channel, convo_id=convo_id, verb='dialog') or self.recall.summaries(service, channel, size=3)
        else:
            text = self.recall.convo(service, channel, convo_id=convo_id, feels=True)

        if not text:
            log.error("∑ summarize_convo: no text")
            return ""

        log.warning("∑ summarizing convo")

        convo_text = '\n'.join(text)

        log.info(convo_text)

        summary = self.completion.get_summary(
            text=convo_text,
            summarizer=f"""
Briefly summarize this dialog, and convert pronouns and verbs to the first person.
Your response must only include the summary and no other text.
""",

        )
        keywords = self.completion.get_keywords(summary)

        if save:
            self.recall.save_summary(service, channel, convo_id, summary, keywords)

        if include_keywords:
            return summary + f"\nKeywords: {keywords}"

        if context_lines:
            return "\n".join(text[-context_lines:] + [summary])

        return summary

    def gather_memories(self, service, channel, entities, visited=None):
        '''
        Look for relevant convos and summaries using memory, relationship graphs, and entity matching.

        TODO: weigh retrievals with "importance" and recency, a la Stanford Smallville
        '''
        if visited is None:
            visited = []

        convo = self.recall.convo(service, channel, feels=False)

        if not convo:
            return visited

        ranked = self.recall.find_related_convos(
            service, channel,
            query='\n'.join(convo[:5]),
            size=10,
            current_convo_id=self.recall.convo_id(service, channel),
            threshold=self.config.memory.relevance
        )

        # No hits? Don't try so hard.
        if not ranked:
            log.warning("🍸 Nothing relevant. Try lateral thinking.")
            ranked = self.recall.find_related_convos(
                service, channel,
                query='\n'.join(convo),
                size=1,
                current_convo_id=self.recall.convo_id(service, channel),
                threshold=self.config.memory.relevance * 1.4,
                any_convo=True
            )

        for hit in ranked:
            if hit.convo_id not in visited:
                if hit.service == 'import_service':
                    log.info("📚 Hit found from import:", hit.channel)
                the_summary = self.recall.get_summary_by_id(hit.convo_id)
                # Hit a sentence? Inject the summary and the sentence.
                if the_summary:
                    self.inject_idea(
                        service, channel,
                        f"{the_summary.summary} In that conversation, {hit.speaker_name} said: {hit.msg}",
                        verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
                    )
                # No summary? Just inject the sentence.
                else:
                    self.inject_idea(
                        service, channel,
                        f"{hit.speaker_name} said: {hit.msg}",
                        verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
                    )
                visited.append(hit.convo_id)
                log.info(f"🧵 Related convo {hit.convo_id} ({float(hit.score):0.3f}):", hit.msg)

        # Look for other summaries that match detected entities
        if entities:
            visited = self.gather_summaries(service, channel, entities, size=2, visited=visited)

        return visited

    def gather_summaries(self, service, channel, entities, size, visited=None):
        '''
        If a previous convo summary matches entities and seems relevant, inject its memory.

        Returns a list of ids of summaries injected.
        '''
        if not entities:
            return []

        if visited is None:
            visited = []

        search_term = ' '.join(entities)
        log.warning(f"ℹ️  look up '{search_term}' in memories")

        for summary in self.recall.summaries(service, channel, search_term, size=10, raw=True):
            if summary.convo_id in visited:
                continue
            visited.append(summary.convo_id)

            log.warning(f"🐘 Memory found: {summary.summary}")
            self.inject_idea(service, channel, summary.summary, "remembers")

            if len(visited) >= size:
                break

        return visited

    def send_chat(self, service, channel, msg):
        '''
        Send a chat message via the autobus.
        '''
        req = {
            "service": service,
            "channel": channel,
            "msg": msg
        }

        try:
            reply = rs.post(f"{self.config.interact.url}/send_msg/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /send_msg/ to interact: {err}")
            return

    def gather_facts(self, service, channel, entities):
        '''
        Gather facts (from Wikipedia) and opinions (from memory).

        This happens asynchronously via the event bus, so facts and opinions
        might not be immediately available for conversation.
        '''
        if not entities:
            return

        the_sample = random.sample(entities, k=min(3, len(entities)))

        req = {
            "service": service,
            "channel": channel,
            "entities": the_sample
        }

        for endpoint in ['opine']:
            log.warning(f"🧾 {endpoint} : {the_sample}")
            try:
                reply = rs.post(f"{self.config.interact.url}/{endpoint}/", params=req, timeout=10)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /{endpoint}/ to interact: {err}")
                return

    def check_goals(self, service, channel, convo):
        ''' Have we achieved our goals? '''
        goals = self.recall.list_goals(service, channel)

        if goals:
            req = {
                "service": service,
                "channel": channel,
                "convo": '\n'.join(convo),
                "goals": goals
            }

            try:
                reply = rs.post(f"{self.config.interact.url}/check_goals/", params=req, timeout=10)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /check_goals/ to interact: {err}")

    def get_feels(self, service, channel, convo_id, room):
        ''' How are we feeling? Let's ask the autobus. '''
        req = {
            "service": service,
            "channel": channel,
            "convo_id": convo_id,
        }
        data = {
            "room": room
        }

        try:
            reply = rs.post(f"{self.config.interact.url}/vibe_check/", params=req, data=data, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /vibe_check/ to interact: {err}")

    def save_knowledge_graph(self, service, channel, convo_id, convo):
        ''' Build a pretty graph of this convo... via the autobus! '''
        req = {
            "service": service,
            "channel": channel,
            "convo_id": convo_id,
        }
        data = {
            "convo": convo
        }

        try:
            reply = rs.post(f"{self.config.interact.url}/build_graph/", params=req, data=data, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /build_graph/ to interact: {err}")

    def retort(self, service, channel, msg, speaker_name, speaker_id, send_chat=True):  # pylint: disable=too-many-locals
        '''
        Get a completion for the given channel.

        Returns the response. If send_chat is True, also send it to chat.
        '''
        log.info(f"💬 get_reply to: {msg}")

        convo_id = self.recall.convo_id(service, channel)
        convo = self.recall.convo(service, channel, feels=False)

        lts = self.recall.get_last_timestamp(service, channel)
        prompt = self.generate_prompt([], convo, service, channel, lts)


        # TODO: vvv  Use this time to backfill context!  vvv

        # Ruminate a bit
        entities = self.extract_entities(msg)

        if entities:
            log.warning(f"🆔 extracted entities: {entities}")
        else:
            entities = self.extract_nouns('\n'.join(convo))[:8]
            log.warning(f"🆔 extracted nouns: {entities}")

        # Reflect on this conversation
        visited = self.gather_memories(service, channel, entities)
        summaries = []
        for doc in self.recall.summaries(service, channel, None, size=1, raw=True):
            if doc.convo_id not in visited and doc.summary not in summaries:
                log.warning("💬 Adding summary:", doc.summary)
                summaries.append(doc.summary)
                visited.append(doc.convo_id)

        # ^^^  end TODO  ^^^

        reply = self.completion.get_reply(prompt)

        if self.custom_filter:
            try:
                reply = self.custom_filter(reply)
            except Exception as err: # pylint: disable=broad-except
                log.warning(f"🤮 Custom filter failed: {err}")

        # Say it!
        if send_chat:
            self.send_chat(service, channel, reply)

        log.info(f"💬 get_reply done: {reply}")

        return reply

    def default_prompt_prefix(self, service, channel):
        ''' The default prompt prefix '''
        ret = [
            f"It is {exact_time()} in the {natural_time()} on {today()}.",
            getattr(self.config.interact, "character", ""),
            f"{self.config.id.name} is feeling {self.recall.feels(self.recall.convo_id(service, channel))}.",
        ]
        goals = self.recall.list_goals(service, channel)
        if goals:
            ret.append(f"{self.config.id.name} is trying to accomplish the following goals: {', '.join(goals)}")
        else:
            log.warning(f"🙅‍♀️ No goal yet for {service} | {channel}")
        return '\n'.join(ret)

    def generate_prompt(self, summaries, convo, service, channel, lts=None):
        ''' Generate the model prompt '''
        newline = '\n'
        timediff = ''
        if lts and elapsed(lts, get_cur_ts()) > 600:
            timediff = f"It has been {ago(lts)} since they last spoke."

        # triples = set()
        graph_summary = ''
        convo_text = '\n'.join(convo)
        # for noun in self.extract_entities(convo_text) + self.extract_nouns(convo_text):
        #     for triple in self.recall.shortest_path(self.recall.bot_name, noun, src_type='Person'):
        #         triples.add(triple)
        # if triples:
        #     graph_summary = self.completion.model.triples_to_text(list(triples))

        # Is this just too much to think about?
        if self.completion.toklen(convo_text + newline.join(summaries)) > self.completion.max_prompt_length():
            log.warning("🥱 generate_prompt(): prompt too long, truncating.")
            convo_text = self.enc.decode(self.enc.encode(convo_text)[:self.completion.max_prompt_length()])

        return f"""{self.default_prompt_prefix(service, channel)}
{newline.join(summaries)}
{graph_summary}
{convo_text}
{timediff}
{self.config.id.name}:"""

    def get_status(self, service, channel):
        ''' status report '''
        return self.generate_prompt(
            self.recall.summaries(service, channel, size=3),
            self.recall.convo(service, channel, feels=True),
            service,
            channel
        )

    def extract_nouns(self, text):
        ''' return a list of all nouns (except pronouns) in text '''
        doc = self.completion.nlp(text)
        nouns = {
            n.text.strip()
            for n in doc.noun_chunks
            if n.text.strip() != self.config.id.name
            for t in n
            if t.pos_ != 'PRON'
        }
        return list(nouns)

    def extract_entities(self, text):
        ''' return a list of all entities in text '''
        doc = self.completion.nlp(text)
        return list({n.text.strip() for n in doc.ents if n.text.strip() != self.config.id.name})

    def inject_idea(self, service, channel, idea, verb="recalls"):
        '''
        Directly inject an idea into recall memory.
        '''
        if verb != "decides" and idea in '\n'.join(self.recall.convo(service, channel, feels=True)):
            log.warning("🤌  Already had this idea, skipping:", idea)
            return

        self.recall.save_convo_line(
            service,
            channel,
            msg=idea,
            speaker_name=self.config.id.name,
            speaker_id=self.config.id.guid,
            convo_id=self.recall.convo_id(service, channel),
            verb=verb
        )

        log.warning(f"🤔 {verb}:", idea)
        return

    def surmise(self, service, channel, topic, size=10):
        ''' Stub for recall '''
        return self.recall.surmise(service, channel, topic, size)

    def add_goal(self, service, channel, goal):
        ''' Stub for recall '''
        return self.recall.add_goal(service, channel, goal)

    def get_goals(self, service, channel, goal=None, size=10):
        ''' Stub for recall '''
        return self.recall.get_goals(service, channel, goal, size)

    def list_goals(self, service, channel, size=10):
        ''' Stub for recall '''
        return self.recall.list_goals(service, channel, size)

    def read_news(self, service, channel, url, title):
        ''' Let's check the news while we ride the autobus. '''
        req = {
            "service": service,
            "channel": channel,
            "url": url,
            "title": title
        }
        try:
            reply = rs.post(f"{self.config.interact.url}/read_news/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /read_news/ to interact: {err}")

    def read_url(self, service, channel, url):
        ''' Let's ride the autobus on the information superhighway. '''
        parsed = urlparse(url)
        if not bool(parsed.scheme) and bool(parsed.netloc):
            log.warning("👨‍💻 Not a URL:", url)
            return

        req = {
            "service": service,
            "channel": channel,
            "url": url
        }
        try:
            reply = rs.post(f"{self.config.interact.url}/read_url/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /read_url/ to interact: {err}")
