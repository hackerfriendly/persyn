'''
interact.py

The limbic system library.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order
import random
import re # used by custom filters

from urllib.parse import urlparse

import requests

# Long and short term memory
from persyn.interaction.memory import Recall

# Time handling
from persyn.interaction.chrono import natural_time, ago, today, elapsed, get_cur_ts

# Prompt completion
from persyn.interaction.completion import LanguageModel

# Color logging
from persyn.utils.color_logging import log


class Interact():
    '''
    The Interact class contains all language handling routines and maintains
    the short and long-term memories for each service+channel.
    '''
    def __init__(self, persyn_config):
        self.config = persyn_config

        # What are we doing with our life?
        self.goals = []

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

    def summarize_convo(
        self,
        service,
        channel,
        save=True,
        max_tokens=200,
        include_keywords=False,
        context_lines=0,
        dialog_only=True,
        model=None
        ):
        '''
        Generate a summary of the current conversation for this channel.
        Also generate and save opinions about detected topics.

        If save == True, save convo to long term memory and generate
        knowledge graph nodes (via the autobus).

        Returns the text summary.
        '''
        convo_id = self.recall.stm.convo_id(service, channel)
        if not convo_id:
            return ""

        if dialog_only:
            text = self.recall.dialog(service, channel) or self.recall.summaries(service, channel, size=3)
        else:
            text = self.recall.convo(service, channel)

        if not text:
            text = [f"{self.config.id.name} isn't sure what is happening."]

        log.warning("∑ summarizing convo")

        convo_text = '\n'.join(text)

        log.info(convo_text)

        summary = self.completion.get_summary(
            text=convo_text,
            summarizer="To briefly summarize this conversation,",
            max_tokens=max_tokens,
            model=model
        )
        keywords = self.completion.get_keywords(summary)

        if save:
            self.recall.summary(service, channel, summary, keywords)
            self.save_knowledge_graph(service, channel, convo_id, convo_text)

        for topic in random.sample(keywords, k=min(3, len(keywords))):
            self.recall.judge(
                service,
                channel,
                topic,
                self.completion.get_opinions(summary, topic),
                convo_id
            )

        if include_keywords:
            return summary + f"\nKeywords: {keywords}"

        if context_lines:
            return "\n".join(text[-context_lines:] + [summary])

        return summary

    def choose_response(self, prompt, convo, service, channel, goals):
        ''' Choose the best completion response from a list of possibilities '''
        if not convo:
            convo = []

        log.info("👍 Choosing a response with model:", self.config.completion.summary_model)
        scored = self.completion.get_replies(
            prompt=prompt,
            convo=convo,
            goals=goals,
            model=self.config.completion.summary_model,
            n=3
        )

        if not scored:
            log.warning("🤨 No surviving replies, try again with model:",
                        self.config.completion.chat_model or self.config.completion.completion_model)
            scored = self.completion.get_replies(
                prompt=prompt,
                convo=convo,
                goals=goals,
                model=self.config.completion.chat_model or self.config.completion.completion_model,
                n=2
            )

        # Uh-oh. Just ignore whatever was last said.
        if not scored:
            log.warning("😳 No surviving replies, one last try with model:", self.config.completion.completion_model)
            scored = self.completion.get_replies(
                prompt=self.generate_prompt([], convo[:-1], service, channel),
                convo=convo,
                goals=goals
            )

        if not scored:
            log.warning("😩 No surviving replies, I give up.")
            log.info("🤷‍♀️ Choice: none available")
            return ":shrug:"

        for item in sorted(scored.items()):
            log.warning(f"{item[0]:0.2f}:", item[1])

        idx = random.choices(list(sorted(scored)), weights=list(sorted(scored)))[0]
        reply = scored[idx]
        log.info(f"✅ Choice: {idx:0.2f}", reply)

        return reply

    def gather_memories(self, service, channel, entities, visited=None):
        '''
        Look for relevant convos and summaries using memory, relationship graphs, and entity matching.
        '''
        if visited is None:
            visited = []

        convo = self.recall.convo(service, channel)

        if not convo:
            return visited


        # TODO: Decide how much convo to use?
        ranked = self.recall.ltm.find_related_convos(
            service, channel,
            convo='\n'.join(convo[:5]),
            size=3,
            current_convo_id=self.recall.stm.convo_id(service, channel),
            threshold=0.3
        )
        # + self.recall.ltm.find_related_convos(
        #     "import_service", "no_channel",
        #     convo=[convo[-1]],
        #     size=1
        # )

        for hit in ranked:
            if hit.convo_id not in visited:
                if hit.service == 'import_service':
                    log.info("📚 Hit found from import:", hit.channel)
                the_summary = self.recall.ltm.get_summary_by_id(hit.convo_id)
                if the_summary:
                    self.inject_idea(
                        service, channel,
                        # This is too expensive. Retrieve old summaries instead.
                        # self.completion.get_summary(hit['hit']['_source']['convo']),
                        the_summary.summary,
                        verb="remembers" # that {ago(hit['@timestamp'])} ago"
                    )
                    visited.append(hit.convo_id)
                    log.info(
                        f"🧵 Related convo {hit.convo_id} ({hit.score}):",
                        f"{the_summary.summary[100:]}"
                    )

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

        for summary in self.recall.lookup_summaries(service, channel, search_term, size=10):
            if summary.convo_id in visited:
                continue
            visited.append(summary.convo_id)

            # # Stay on topic
            # prompt = '\n'.join(
            #     self.recall.convo(service, channel)
            #     + [
            #         f"{self.config.id.name} remembers that {ago(summary['_source']['@timestamp'])} ago: "
            #         + summary['_source']['summary']
            #     ]
            # )
            # on_topic = self.completion.get_summary(
            #     prompt,
            #     summarizer="Q: True or False: this memory relates to the earlier conversation.\nA:",
            #     max_tokens=10)

            # log.warning(f"🧐 Are we on topic? {on_topic}")
            # if 'true' not in on_topic.lower():
            #     log.warning(f"🚫 Irrelevant memory discarded: {summary['_source']['summary']}")
            #     continue

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
            reply = requests.post(f"{self.config.interact.url}/send_msg/", params=req, timeout=10)
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

        for endpoint in ('opine', 'wikipedia'):
            try:
                reply = requests.post(f"{self.config.interact.url}/{endpoint}/", params=req, timeout=10)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /{endpoint}/ to interact: {err}")
                return

    def check_goals(self, service, channel, convo):
        ''' Have we achieved our goals? '''
        self.goals = self.recall.list_goals(service, channel)

        if self.goals:
            req = {
                "service": service,
                "channel": channel,
                "convo": '\n'.join(convo),
                "goals": self.goals
            }

            try:
                reply = requests.post(f"{self.config.interact.url}/check_goals/", params=req, timeout=10)
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
            reply = requests.post(f"{self.config.interact.url}/vibe_check/", params=req, data=data, timeout=10)
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
            reply = requests.post(f"{self.config.interact.url}/build_graph/", params=req, data=data, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /build_graph/ to interact: {err}")

    # Need to instrument this. It takes far too long and isn't async.
    def get_reply(self, service, channel, msg, speaker_name, speaker_id):  # pylint: disable=too-many-locals
        '''
        Get the best reply for the given channel. Saves to recall memory.

        Returns the best available reply.
        '''
        self.goals = self.recall.list_goals(service, channel)

        # This should be async, separate thread?
        if self.recall.expired(service, channel):
            self.summarize_convo(service, channel, save=True, context_lines=2)

        if msg != '...':
            self.recall.save(service, channel, msg, speaker_name, speaker_id, verb='dialog')

        convo = self.recall.convo(service, channel)
        last_sentence = None

        if convo:
            last_sentence = convo.pop()

        # This should be async, separate thread?
        # Save the knowledge graph every 5 lines
        if convo and len(convo) % 5 == 0:
            self.save_knowledge_graph(service, channel, self.recall.stm.convo_id(service, channel), convo)

        # Ruminate a bit
        entities = self.extract_entities(msg)

        if entities:
            log.warning(f"🆔 extracted entities: {entities}")
        else:
            entities = self.extract_nouns('\n'.join(convo))[:8]
            log.warning(f"🆔 extracted nouns: {entities}")

        # Reflect on this conversation
        visited = self.gather_memories(service, channel, entities)

        # Facts and opinions
        self.gather_facts(service, channel, entities)

        # This should be async, separate thread?
        # Also, where did the goals go? Haven't seen a trophy in ages.
        # Goals. Don't give out _too_ many trophies.
        if random.random() < 0.5:
            self.check_goals(service, channel, convo)

        # Our mind might have been wandering, so remember the last thing that was said.
        if last_sentence:
            convo.append(last_sentence)

        summaries = []
        for summary in self.recall.lookup_summaries(service, channel, None, size=5):
            if summary.convo_id not in visited and summary.summary not in summaries:
                summaries.append(summary.summary)
                visited.append(summary.convo_id)

        lts = self.recall.lts(service, channel)
        prompt = self.generate_prompt(summaries, convo, service, channel, lts)

        # Is this just too much to think about?
        if self.completion.toklen(prompt) > self.completion.max_prompt_length():
            # Kick off a summary request via autobus. Yes, we're talking to ourselves now.
            log.warning("🥱 get_reply(): prompt too long, summarizing.")
            req = {
                "service": service,
                "channel": channel,
                "save": True,
                "max_tokens": 100
            }
            try:
                reply = requests.post(f"{self.config.interact.url}/summary/", params=req, timeout=60)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /summary/ to interact: {err}")
                return " :dancer: :interrobang: "

            prompt = self.generate_prompt([], convo, service, channel, lts)

        reply = self.choose_response(prompt, convo, service, channel, self.goals)
        if self.custom_filter:
            try:
                reply = self.custom_filter(reply)
            except Exception as err: # pylint: disable=broad-except
                log.warning(f"🤮 Custom filter failed: {err}")

        # Say it!
        self.send_chat(service, channel, reply)

        # Sentiment analysis via the autobus
        self.get_feels(service, channel, self.recall.stm.convo_id(service, channel), f'{prompt} {reply}')

        if 'http' in msg:
            # Regex chosen by GPT-4. 😵‍💫
            for url in re.findall(r'http[s]?://(?:[^\s()<>\"\']|(?:\([^\s()<>]*\)))+', msg):
                self.read_url(service, channel, url)

        return reply

    def default_prompt_prefix(self, service, channel):
        ''' The default prompt prefix '''
        ret = [
            f"It is {natural_time()} on {today()}.",
            getattr(self.config.interact, "character", ""),
            f"{self.config.id.name} is feeling {self.recall.feels(self.recall.stm.convo_id(service, channel))}.",
        ]
        if self.goals:
            ret.append(f"{self.config.id.name} is trying to accomplish the following goals: {', '.join(self.goals)}")
        return '\n'.join(ret)

    def generate_prompt(self, summaries, convo, service, channel, lts=None):
        ''' Generate the model prompt '''
        newline = '\n'
        timediff = ''
        if lts and elapsed(lts, get_cur_ts()) > 600:
            timediff = f"It has been {ago(lts)} since they last spoke."

        triples = set()
        graph_summary = ''
        convo_text = '\n'.join(convo)
        for noun in self.extract_entities(convo_text) + self.extract_nouns(convo_text):
            for triple in self.recall.ltm.shortest_path(self.recall.bot_name, noun, src_type='Person'):
                triples.add(triple)
        if triples:
            graph_summary = self.completion.model.triples_to_text(list(triples))

        return f"""{self.default_prompt_prefix(service, channel)}
{newline.join(summaries)}
{graph_summary}
{convo_text}
{timediff}
{self.config.id.name}:"""

    def get_status(self, service, channel):
        ''' status report '''
        return self.generate_prompt(
            self.recall.summaries(service, channel, size=2),
            self.recall.convo(service, channel),
            service,
            channel
        )

    def amnesia(self, service, channel):
        ''' forget it '''
        return self.recall.forget(service, channel)

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
        if verb != "decides" and idea in '\n'.join(self.recall.convo(service, channel)):
            log.warning("🤌  Already had this idea, skipping:", idea)
            return

        if self.recall.expired(service, channel):
            self.summarize_convo(service, channel, save=True, context_lines=2)

        self.recall.save(service, channel, idea, self.config.id.name, self.config.id.guid, verb)

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
            reply = requests.post(f"{self.config.interact.url}/read_news/", params=req, timeout=10)
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
            reply = requests.post(f"{self.config.interact.url}/read_url/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /read_url/ to interact: {err}")
