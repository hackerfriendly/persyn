'''
interact.py

The limbic system library.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order
import json
import random
import re # used by custom filters
import urllib3

from spacy.lang.en.stop_words import STOP_WORDS

# just-in-time Wikipedia
import wikipedia
from wikipedia.exceptions import WikipediaException

from Levenshtein import ratio

# Long and short term memory
from interaction.memory import Recall

# Time handling
from interaction.chrono import natural_time, ago, today, elapsed, get_cur_ts

# Prompt completion
from interaction.completion import LanguageModel

# Color logging
from utils.color_logging import log


class Interact():
    '''
    The Interact class contains all language handling routines and maintains
    the short and long-term memories for each service+channel.
    '''
    def __init__(self, persyn_config):
        self.config = persyn_config

        # local Wikipedia cache
        self.wikicache = {}

        # How are we feeling today? TODO: This needs to be per-channel, particularly the goals.
        self.feels = {'current': "nothing in particular", 'goals': []}

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

        # Elasticsearch memory:
        # First, check if we don't want to verify TLS certs (because self-hosted Elasticsearch)
        verify_certs_setting = persyn_config.memory.elastic.get("verify_certs", "true")
        verify_certs = json.loads(str(verify_certs_setting).lower()) # convert "false" -> False, "0" -> False

        # If not, disable the pesky urllib3 insecure request warning.
        if not verify_certs:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        # Then create the Recall object using the Elasticsearch credentials.
        self.recall = Recall(
            bot_name=persyn_config.id.name,
            bot_id=persyn_config.id.guid,
            url=persyn_config.memory.elastic.url,
            auth_name=persyn_config.memory.elastic.user,
            auth_key=persyn_config.memory.elastic.key,
            index_prefix=persyn_config.memory.elastic.index_prefix,
            conversation_interval=600, # ten minutes
            verify_certs=verify_certs
        )

    def summarize_convo(self, service, channel, save=True, max_tokens=200, include_keywords=False, context_lines=0):
        '''
        Generate a summary of the current conversation for this channel.
        Also generate and save opinions about detected topics.
        If save == True, save it to long term memory.
        Returns the text summary.
        '''

        convo_id = self.recall.stm.convo_id(service, channel)
        if not convo_id:
            return ""

        log.warning("∑ saving relationships")
        self.recall.ltm.save_relationship_graph(
            service, channel,
            convo_id,
            ' '.join(self.recall.convo(service, channel))
        )

        dialog = self.recall.dialog(service, channel) or self.recall.summaries(service, channel, size=3)
        if not dialog:
            dialog = [f"{self.config.id.name} isn't sure what is happening."]

        log.warning("∑ summarizing convo")

        summary = self.completion.get_summary(
            text='\n'.join(dialog),
            summarizer="To briefly summarize this conversation,",
            max_tokens=max_tokens
        )
        keywords = self.completion.get_keywords(summary)

        if save:
            self.recall.summary(service, channel, summary, keywords)

        for topic in random.sample(keywords, k=min(3, len(keywords))):
            self.recall.judge(
                service,
                channel,
                topic,
                self.completion.get_opinions(summary, topic)
            )

        if include_keywords:
            return summary + f"\nKeywords: {keywords}"

        if context_lines:
            return "\n".join(dialog[-context_lines:] + [summary])

        return summary

    def choose_response(self, prompt, convo, goals):
        ''' Choose the best completion response from a list of possibilities '''
        if not convo:
            convo = []

        scored = self.completion.get_replies(
            prompt=prompt,
            convo=convo,
            goals=goals
        )

        if not scored:
            log.warning("🤨 No surviving replies, try again.")
            scored = self.completion.get_replies(
                prompt=prompt,
                convo=convo,
                goals=goals
            )

        # Uh-oh. Just keep it brief.
        if not scored:
            log.warning("😳 No surviving replies, one last try.")
            scored = self.completion.get_replies(
                prompt=self.generate_prompt([], convo[-4:]),
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
        Look for relevant convos and summaries using elasticsearch, relationship graphs, and entity matching.
        '''
        if visited is None:
            visited = []

        # Use the entire existing convo first
        ranked = self.recall.find_related_convos(service, channel, size=3)
        for hit in ranked:
            hit_id = hit['hit'].get('convo_id', hit['hit']['_id'])
            if hit_id not in visited:
                self.inject_idea(
                    service, channel,
                    self.completion.get_summary(hit['hit']['_source']['convo']),
                    verb=f"remembers that {ago(hit['hit']['_source']['@timestamp'])} ago"
                )
                visited.append(hit_id)
                log.info(
                    f"🧵 Related relationship {hit_id} ({hit['score']}):",
                    f"{hit['hit']['_source']['convo'][:100]}..."
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
            if summary['_id'] in visited:
                continue
            visited.append(summary['_id'])

            # Stay on topic
            prompt = '\n'.join(
                self.recall.convo(service, channel)
                + [
                    f"{self.config.id.name} remembers that {ago(summary['_source']['@timestamp'])} ago: "
                    + summary['_source']['summary']
                ]
            )
            on_topic = self.completion.get_summary(
                prompt,
                summarizer="Q: True or False: this memory relates to the earlier conversation.\nA:",
                max_tokens=10)

            log.warning(f"🧐 Are we on topic? {on_topic}")
            if 'true' not in on_topic.lower():
                log.warning(f"🚫 Irrelevant memory discarded: {summary['_source']['summary']}")
                continue

            log.warning(f"🐘 Memory found: {summary['_source']['summary']}")
            self.inject_idea(service, channel, summary['_source']['summary'], "remembers")

            if len(visited) >= size:
                break

        return visited

    def gather_facts(self, service, channel, entities):
        ''' Gather facts (from Wikipedia) and opinions (from memory) '''
        if not entities:
            return

        for entity in random.sample(entities, k=min(3, len(entities))):
            if entity == '' or entity in STOP_WORDS:
                continue

            opinions = self.recall.opine(service, channel, entity)
            if opinions:
                log.warning(f"🙋‍♂️ Opinions about {entity}: {len(opinions)}")
                if len(opinions) == 1:
                    opinion = opinions[0]
                else:
                    opinion = self.completion.nlp(self.completion.get_summary(
                        text='\n'.join(opinions),
                        summarizer=f"{self.config.id.name}'s opinion about {entity} can be briefly summarized as:",
                        max_tokens=75
                    )).text

                if opinion not in self.recall.stm.get_bias(service, channel):
                    self.recall.stm.add_bias(service, channel, opinion)
                    self.inject_idea(service, channel, opinion, f"thinks about {entity}")

            log.warning(f'❇️  look up "{entity}" on Wikipedia')

            if entity in self.wikicache:
                log.warning(f'🤑 wiki cache hit: "{entity}"')
            else:
                wiki = None
                try:
                    if wikipedia.page(entity).original_title.lower() != entity.lower():
                        log.warning("❎ no exact match found")
                        continue

                    log.warning("✅ found it.")
                    wiki = wikipedia.summary(entity, sentences=3)

                    summary = self.completion.nlp(self.completion.get_summary(
                        text=f"This Wikipedia article:\n{wiki}",
                        summarizer="Can be briefly summarized as: ",
                        max_tokens=75
                    ))
                    # 2 sentences max please.
                    self.wikicache[entity] = ' '.join([s.text for s in summary.sents][:2])

                except WikipediaException:
                    log.warning("❎ no unambigous wikipedia entry found")
                    self.wikicache[entity] = None
                    continue

                if entity in self.wikicache and self.wikicache[entity] is not None:
                    self.inject_idea(service, channel, self.wikicache[entity])

    def check_goals(self, service, channel, convo):
        ''' Have we achieved our goals? '''
        achieved = []

        if self.feels['goals']:
            goal = random.choice(self.feels['goals'])
            goal_achieved = self.completion.get_summary(
                '\n'.join(convo),
                summarizer=f"Q: True or False: {self.config.id.name} achieved the goal of {goal}.\nA:",
                max_tokens=10
            )

            log.warning(f"🧐 Did we achieve our goal? {goal_achieved}")
            if 'true' in goal_achieved.lower():
                log.warning(f"🏆 Goal achieved: {goal}")
                achieved.append(goal)
                self.feels['goals'].remove(goal)
            else:
                log.warning(f"🚫 Goal not yet achieved: {goal}")

        summary = self.completion.nlp(self.completion.get_summary(
            text='\n'.join(convo),
            summarizer=f"{self.config.id.name}'s goal is",
            max_tokens=100
        ))

        # 1 sentence max please.
        the_goal = ' '.join([s.text for s in summary.sents][:1])

        # some goals are too easy
        for taboo in ['remember', 'learn']:
            if taboo in the_goal:
                return achieved

        # don't repeat yourself
        for goal in self.recall.get_goals(service, channel):
            if ratio(goal, the_goal) > 0.6:
                return achieved

        # we've been handing out too many trophies
        if random.random() < 0.5:
            self.recall.add_goal(service, channel, the_goal)

        return achieved

    def get_reply(self, service, channel, msg, speaker_name, speaker_id): # pylint: disable=too-many-locals
        '''
        Get the best reply for the given channel. Saves to recall memory.

        Returns the best reply, and any goals achieved.
        '''
        self.feels['goals'] = self.recall.get_goals(service, channel)

        if self.recall.expired(service, channel):
            self.summarize_convo(service, channel, save=True, context_lines=2)

        if msg != '...':
            self.recall.save(service, channel, msg, speaker_name, speaker_id, verb='dialog')

        convo = self.recall.convo(service, channel)
        last_sentence = None

        if convo:
            last_sentence = convo.pop()

        # Ruminate a bit
        entities = self.extract_entities(msg)

        if entities:
            log.warning(f"🆔 extracted entities: {entities}")
        else:
            entities = self.completion.get_keywords(convo)
            log.warning(f"🆔 extracted keywords: {entities}")

        # Reflect on this conversation
        visited = self.gather_memories(service, channel, entities)

        # Facts and opinions (interleaved)
        self.gather_facts(service, channel, entities)

        # Goals
        achieved = self.check_goals(service, channel, convo)

        # Our mind might have been wandering, so remember the last thing that was said.
        if last_sentence:
            convo.append(last_sentence)

        summaries = []
        for summary in self.recall.lookup_summaries(service, channel, None, size=5):
            if summary['_id'] not in visited and summary['_source']['summary'] not in summaries:
                summaries.append(summary['_source']['summary'])
                visited.append(summary['_id'])

        lts = self.recall.lts(service, channel)
        prompt = self.generate_prompt(summaries, convo, lts)

        # Is this just too much to think about?
        if len(prompt) > self.completion.max_prompt_length:
            log.warning("🥱 get_reply(): prompt too long, summarizing.")
            self.summarize_convo(service, channel, save=True, max_tokens=100)
            summaries = self.recall.summaries(service, channel, size=3)
            prompt = self.generate_prompt(summaries, convo[-3:], lts)

        reply = self.choose_response(prompt, convo, self.feels['goals'])
        if self.custom_filter:
            try:
                reply = self.custom_filter(reply)
            except Exception as err: # pylint: disable=broad-except
                log.warning(f"🤮 Custom filter failed: {err}")

        self.recall.save(service, channel, reply, self.config.id.name, self.config.id.guid, verb='dialog')

        self.feels['current'] = self.completion.get_feels(f'{prompt} {reply}')

        log.warning("😄 Feeling:", self.feels['current'])

        return (reply, achieved)

    def default_prompt_prefix(self):
        ''' The default prompt prefix '''
        return '\n'.join([
            getattr(self.config.interact, "character", ""),
            f"It is {natural_time()} on {today()}.",
            f"{self.config.id.name} is feeling {self.feels['current']}.",
            f"{self.config.id.name}'s goals include {', '.join(self.feels['goals'])}" if self.feels['goals'] else ''
        ])

    def generate_prompt(self, summaries, convo, lts=None):
        ''' Generate the model prompt '''
        newline = '\n'
        timediff = ''
        if lts and elapsed(lts, get_cur_ts()) > 600:
            timediff = f"It has been {ago(lts)} since they last spoke."

        return f"""{self.default_prompt_prefix()}
{newline.join(summaries)}
{newline.join(convo)}
{timediff}
{self.config.id.name}:"""

    def get_status(self, service, channel):
        ''' status report '''
        paragraph = '\n\n'
        newline = '\n'
        summaries = self.recall.summaries(service, channel, size=2)
        convo = self.recall.convo(service, channel)
        timediff = f"It has been {ago(self.recall.lts(service, channel))} since they last spoke."
        return f"""{self.default_prompt_prefix()}
{paragraph.join(summaries)}

{newline.join(convo)}
{timediff}
"""

    def amnesia(self, service, channel):
        ''' forget it '''
        return self.recall.forget(service, channel)

    def extract_nouns(self, text):
        ''' return a list of all nouns (except pronouns) in text '''
        nlp = self.completion.nlp(text)
        nouns = {
            n.text.strip()
            for n in nlp.noun_chunks
            if n.text.strip() != self.config.id.name
            for t in n
            if t.pos_ != 'PRON'
        }
        return list(nouns)

    def extract_entities(self, text):
        ''' return a list of all entities in text '''
        nlp = self.completion.nlp(text)
        return list({n.text.strip() for n in nlp.ents if n.text.strip() != self.config.id.name})

    def daydream(self, service, channel):
        ''' Chew on recent conversation '''
        paragraph = '\n\n'
        newline = '\n'
        summaries = self.recall.summaries(service, channel, size=5)
        convo = self.recall.convo(service, channel)

        reply = {}
        entities = self.extract_entities(paragraph.join(summaries) + newline.join(convo))

        for entity in random.sample(entities, k=min(3, len(entities))):
            if entity == '' or entity in STOP_WORDS:
                continue

            if entity in self.wikicache:
                log.warning(f"🤑 wiki cache hit: {entity}")
                reply[entity] = self.wikicache[entity]
            else:
                try:
                    hits = wikipedia.search(entity)
                    if hits:
                        try:
                            wiki = wikipedia.summary(hits[0:1], sentences=3)
                            summary = self.completion.nlp(self.completion.get_summary(
                                text=f"This Wikipedia article:\n{wiki}",
                                summarizer="Can be summarized as: ",
                                max_tokens=100
                            ))
                            # 2 sentences max please.
                            reply[entity] = ' '.join([s.text for s in summary.sents][:2])
                            self.wikicache[entity] = reply[entity]
                        except WikipediaException:
                            continue

                except WikipediaException:
                    continue

        log.warning("💭 daydream entities:")
        log.warning(reply)
        return reply

    def inject_idea(self, service, channel, idea, verb="recalls"):
        '''
        Directly inject an idea into recall memory.
        '''
        if self.recall.expired(service, channel):
            self.summarize_convo(service, channel, save=True, context_lines=2)

        self.recall.save(service, channel, idea, self.config.id.name, self.config.id.guid, verb)

        log.warning(f"🤔 {verb}:", idea)
        return "🤔"

    def opine(self, service, channel, entity, speaker_id=None, size=10):
        ''' Stub for recall '''
        return self.recall.opine(service, channel, entity, speaker_id, size)

    def add_goal(self, service, channel, goal):
        ''' Stub for recall '''
        return self.recall.add_goal(service, channel, goal)

    def get_goals(self, service, channel):
        ''' Stub for recall '''
        return self.recall.get_goals(service, channel)
