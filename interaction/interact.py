'''
interact.py

The limbic system library.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order
import json
import random
import sys

from pathlib import Path

from spacy.lang.en.stop_words import STOP_WORDS

# just-in-time Wikipedia
import wikipedia
from wikipedia.exceptions import WikipediaException

from Levenshtein import ratio

# Add persyn root to sys.path
sys.path.insert(0, str((Path(__file__) / '../../').resolve()))

# text-to-speech
# from voice import tts

# Long and short term memory
from memory import Recall

# Time handling
from chrono import natural_time, ago, today, elapsed, get_cur_ts

# Prompt completion
from completion import LanguageModel

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

        # Elasticsearch memory:
        # First, check if we don't want to verify TLS certs (because self-hosted Elasticsearch)
        verify_certs_setting = persyn_config.memory.elastic.get("verify_certs", "true")
        verify_certs = json.loads(str(verify_certs_setting).lower()) # convert "false" -> False, "0" -> False

        # If not, disable the pesky urllib3 insecure request warning.
        if not verify_certs:
            import urllib3
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

    def dialog(self, convo):
        '''
        Return only the words actually spoken in a convo
        '''
        log.critical(convo)
        ret = []
        for c in convo:
            if c['speaker'].endswith(" remembers") or c['speaker'].endswith(" recalls") or c['speaker'].endswith(" thinks") or c['speaker'].endswith(" posted"):
                continue
            ret.append(f"{c['speaker']}: {c['msg']}")

        return ret

    def summarize_convo(self, service, channel, save=True, max_tokens=200, include_keywords=False, context_lines=0):
        '''
        Generate a summary of the current conversation for this channel.
        Also generate and save opinions about detected topics.
        If save == True, save it to long term memory.
        Returns the text summary.
        '''
        summaries, convo_, _ = self.recall.load(service, channel, summaries=0)
        if convo_:
            convo = [f"{c['speaker']}: {c['msg']}" for c in convo_]
        else:
            summaries, convo, _ = self.recall.load(service, channel, summaries=3)
            if not summaries:
                summaries = [ f"{self.config.id.name} isn't sure what is happening." ]

            # No convo? summarize the summaries
            convo = summaries

        spoken = self.dialog(convo_)
        log.warning(f"∑ summarizing convo: {json.dumps(spoken)}")

        summary = self.completion.get_summary(
            text='\n'.join(convo),
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
            return "\n".join(convo[-context_lines:] + [summary])

        return summary

    def choose_reply(self, prompt, convo, goals):
        ''' Choose the best reply from a list of possibilities '''
        if not convo:
            # convo = [f'{self.config.id.name} changes the subject.']
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

    def gather_memories(self, service, channel, entities, summaries, convo):
        ''' Take a trip down memory lane '''
        search_term = ' '.join(entities)
        log.warning(f"ℹ️  look up '{search_term}' in memories")

        for memory in self.recall.remember(service, channel, search_term, summaries=5):
            # Don't repeat yourself, loopy-lou.
            if memory['text'] in summaries or memory['text'] in '\n'.join(convo):
                continue

            # Stay on topic
            prompt = '\n'.join(
                convo[:-1]
                + [f"{self.config.id.name} remembers that {ago(memory['timestamp'])} ago: "
                + memory['text']]
            )
            on_topic = self.completion.get_summary(
                prompt,
                summarizer="Q: True or False: this memory relates to the earlier conversation.\nA:",
                max_tokens=10)

            log.warning(f"🧐 Are we on topic? {on_topic}")
            if 'true' not in on_topic.lower():
                log.warning(f"🚫 Irrelevant memory discarded: {memory}")
                continue

            log.warning(f"🐘 Memory found: {memory}")
            self.inject_idea(service, channel, memory['text'], f"remembers that {ago(memory['timestamp'])} ago")
            break

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
            self.recall.save(service, channel, msg, speaker_name, speaker_id)
            # tts(msg)

        # Load summaries and conversation
        summaries, convo_, lts = self.recall.load(service, channel, summaries=2)
        convo = [f"{c['speaker']}: {c['msg']}" for c in convo_]
        convo_length = len(convo)
        last_sentence = None

        if convo:
            last_sentence = convo[-1]

        # Ruminate a bit
        entities = self.extract_entities(msg)

        if entities:
            log.warning(f"🆔 extracted entities: {entities}")
        else:
            entities = self.completion.get_keywords(convo)
            log.warning(f"🆔 extracted keywords: {entities}")

        if entities:
            # Memories
            self.gather_memories(service, channel, entities, summaries, convo)

            # Facts and opinions (interleaved)
            self.gather_facts(service, channel, entities)

        # Goals
        achieved = self.check_goals(service, channel, convo)

        # If our mind was wandering, remember the last thing that was said.
        if convo_length != len(self.recall.load(service, channel, summaries=0)[1]):
            self.inject_idea(service, channel, last_sentence)

        prompt = self.generate_prompt(summaries, convo, lts)

        # Is this just too much to think about?
        if len(prompt) > self.completion.max_prompt_length:
            log.warning("🥱 get_reply(): prompt too long, summarizing.")
            self.summarize_convo(service, channel, save=True, max_tokens=100)
            summaries, _, _ = self.recall.load(service, channel, summaries=3)
            prompt = self.generate_prompt(summaries, convo[-3:], lts)

        reply = self.choose_reply(prompt, convo, self.feels['goals'])

        self.recall.save(service, channel, reply, self.config.id.name, self.config.id.guid)

        # tts(reply, voice=self.config.voice.personality)

        self.feels['current'] = self.completion.get_feels(f'{prompt} {reply}')

        log.warning("😄 Feeling:", self.feels['current'])

        return (reply, achieved)

    def default_prompt_prefix(self):
        ''' The default prompt prefix '''
        if self.feels['goals']:
            goals = f"""\n{self.config.id.name}'s goals include {', '.join(self.feels['goals'])}."""
        else:
            goals = ""

        return f"""It is {natural_time()} on {today()}. {self.config.id.name} is feeling {self.feels['current']}.{goals}"""

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
        summaries, convo_, lts = self.recall.load(service, channel, summaries=2)
        convo = [f"{c['speaker']}: {c['msg']}" for c in convo_]
        timediff = f"It has been {ago(lts)} since they last spoke."
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
        summaries, convo_, _ = self.recall.load(service, channel, summaries=5)
        convo = [f"{c['speaker']}: {c['msg']}" for c in convo_]

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
        ''' Directly inject an idea into recall memory. '''
        if self.recall.expired(service, channel):
            self.summarize_convo(service, channel, save=True, context_lines=2)

        self.recall.save(service, channel, idea, f"{self.config.id.name} {verb}", self.config.id.guid)

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
