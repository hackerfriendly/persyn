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
        convo_id = self.recall.convo_id(service, channel)
        if not convo_id:
            return ""

        if dialog_only:
            text = self.recall.convo(service, channel, verb='dialog') or self.recall.lookup_summaries(service, channel, size=3)
        else:
            text = self.recall.convo(service, channel)

        if not text:
            return random.choice([
                f"{self.config.id.name} feels mystified by the current state of affairs.",
                f"{self.config.id.name} is bewildered by the present circumstances.",
                f"{self.config.id.name} is finding the present situation puzzling and hard to comprehend.",
                f"{self.config.id.name} is in a haze of confusion about what's happening.",
                f"{self.config.id.name} is in the dark about the ongoing situation.",
                f"{self.config.id.name} is struggling to make sense of the ongoing situation.",
                f"{self.config.id.name} isn't sure what is happening.",
                f"{self.config.id.name}, ordinarily quick to comprehend, is genuinely befuddled by the current state of things.",
                f"{self.config.id.name}, usually in tune with their surroundings, is completely at sea with what's unfolding.",
                f"{self.config.id.name}, usually self-reliant and unshakeable, is grappling with ambiguity regarding the present scenario.",
                f"{self.config.id.name}, usually up-to-date and aware, is strangely oblivious to the current scenario.",
                f"Despite their sharp intuition, {self.config.id.name} is clueless about the present events.",
                f"Despite their usual firm grasp and assurance, {self.config.id.name} is confronting a cloud of uncertainty.",
                f"Despite their usual perceptiveness, {self.config.id.name} is struggling to grasp the details of the ongoing situation.",
                f"Even with their keen insight, {self.config.id.name} is in the dark about the ongoing developments.",
                f"Even with their sharp wits, {self.config.id.name} is unable to decode the present circumstances.",
                f"The current situation has put {self.config.id.name}, who is usually unflappable, in a state of confusion.",
                f"The existing circumstances have left {self.config.id.name} perplexed.",
                f"The present context has thrown {self.config.id.name} into a sphere of uncertainty.",
                f"Typically informed, {self.config.id.name} is out of the loop regarding the present situation.",
                f"Usually quick on the uptake, {self.config.id.name} seems lost in the fog of the current events.",
            ])

        log.warning("∑ summarizing convo")

        convo_text = '\n'.join(text)

        log.info(convo_text)

        summary = self.completion.get_summary(
            text=convo_text,
            summarizer=f"Briefly summarize this conversation from {self.config.id.name}'s point of view, and convert pronouns and verbs to the first person.",
            max_tokens=max_tokens,
            model=model
        )
        keywords = self.completion.get_keywords(summary)

        if save:
            self.recall.save_summary(service, channel, convo_id, summary, keywords)
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

        ranked = self.recall.find_related_convos(
            service, channel,
            convo='\n'.join(convo[:3]),
            size=10,
            current_convo_id=self.recall.convo_id(service, channel),
            threshold=0.15
        ) + self.recall.find_related_convos(
            service, channel,
            convo='\n'.join(convo),
            size=2,
            current_convo_id=self.recall.convo_id(service, channel),
            threshold=0.2
        )

        for hit in ranked:
            if hit.convo_id not in visited:
                if hit.service == 'import_service':
                    log.info("📚 Hit found from import:", hit.channel)
                the_summary = self.recall.get_summary_by_id(hit.convo_id)
                if the_summary:
                    self.inject_idea(
                        service, channel,
                        f"{the_summary.summary} In that conversation, {hit.speaker_name} said: {hit.msg}",
                        verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
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
        goals = self.recall.list_goals(service, channel)

        if goals:
            req = {
                "service": service,
                "channel": channel,
                "convo": '\n'.join(convo),
                "goals": goals
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
        log.info(f"💬 get_reply to: {msg}")

        # goals = self.recall.list_goals(service, channel)

        convo_id = self.recall.convo_id(service, channel)

        if msg != '...':
            self.recall.save_convo_line(
                service,
                channel,
                msg,
                speaker_name,
                speaker_id,
                convo_id=convo_id,
                verb='dialog'
            )

        convo = self.recall.convo(service, channel)
        last_sentence = None

        if convo:
            last_sentence = convo.pop()

        # This should be async, separate thread?
        # Save the knowledge graph every 5 lines
        if convo and len(convo) % 5 == 0:
            self.save_knowledge_graph(service, channel, convo_id, convo)

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

        lts = self.recall.get_last_timestamp(service, channel)
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

        reply = self.choose_response(prompt, convo, service, channel, self.recall.list_goals(service, channel))
        if self.custom_filter:
            try:
                reply = self.custom_filter(reply)
            except Exception as err: # pylint: disable=broad-except
                log.warning(f"🤮 Custom filter failed: {err}")

        # Say it!
        self.send_chat(service, channel, reply)

        # Sentiment analysis via the autobus
        self.get_feels(service, channel, convo_id, f'{prompt} {reply}')

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

        triples = set()
        graph_summary = ''
        convo_text = '\n'.join(convo)
        for noun in self.extract_entities(convo_text) + self.extract_nouns(convo_text):
            for triple in self.recall.shortest_path(self.recall.bot_name, noun, src_type='Person'):
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
            self.recall.lookup_summaries(service, channel, size=2),
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
