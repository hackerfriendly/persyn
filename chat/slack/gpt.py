''' GPT-3 completion engine '''
import os
import random
import re

from collections import Counter

import openai
import spacy

from ftfy import fix_text

from feels import get_flair_score, get_feels_score, get_profanity_score

# Color logging
from color_logging import debug, info, warning, error, critical # pylint: disable=unused-import

class GPT():
    ''' Container for GPT-3 completion requests '''
    def __init__(
        self,
        bot_name,
        min_score=0.0,
        api_key=os.getenv('OPENAI_API_KEY'),
        engine=os.environ.get('OPENAI_MODEL', 'davinci')
        ):
        self.bot_name = bot_name
        self.min_score = min_score
        self.engine = engine
        self.stats = Counter()
        self.nlp = spacy.load("en_core_web_lg")
        openai.api_key = api_key

        # Strictly forbidden words
        self.forbidden = ['Elsa', 'Arendelle', 'Kristoff', 'Olaf', 'Frozen']

    def get_best_reply(self, prompt, convo, stop=None, temperature=0.9, max_tokens=200):
        ''' Given a text prompt and recent conversation, send the prompt to GPT3 and return a simple text response. '''
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=8,
            frequency_penalty=0.5,
            presence_penalty=0.6,
            stop=stop
        )
        # debug(f'🧠 Prompt: {prompt}\n🗣 Response:', response)

        # Choose a response based on the most positive sentiment.
        scored = self.score_choices(response.choices, convo)

        if not scored:
            self.stats.update(['replies exhausted'])
            return ':shrug:'

        for item in sorted(scored.items()):
            warning(f"{item[0]:0.2f}:", item[1])

        idx = random.choices(list(sorted(scored)), weights=list(sorted(scored)))[0]
        reply = scored[idx]

        warning(f"✅ Choice: {idx:0.2f}", reply)
        warning("📊 Stats:", self.stats)

        return reply

    def truncate(self, text):
        '''
        Extract the first few "sentences" from OpenAI's messy output.
        ftfy.fix_text() fixes encoding issues and replaces fancy quotes with ascii.
        spacy parses sentence structure.
        '''
        doc = self.nlp(fix_text(text))
        sents = list(doc.sents)
        # Always take the first "sentence"
        reply = [cleanup(sents[0].text)]
        # Possibly add more
        try:
            for sent in sents[1:4]:
                if ':' in sent.text:
                    return ' '.join(reply)
                re.search('[a-zA-Z]', sent.text)
                if not any(c.isalpha() for c in sent.text):
                    continue

                reply.append(cleanup(sent.text))

            return ' '.join(reply)
        except TypeError:
            return ''

    def score_choices(self, choices, convo):
        '''
        Filter potential responses for quality, sentimentm and profanity.
        Rank the remaining choices by sentiment and return the ranked list of possible choices.
        '''
        scored = {}

        nouns_in_convo = {word.lemma_ for word in self.nlp(' '.join(convo)) if word.pos_ == "NOUN"}

        for choice in choices:
            # Only consider the first line
            text = self.truncate(choice['text'])

            # Skip blanks
            if not text:
                self.stats.update(['blank'])
                continue
            # No urls
            if 'http' in text:
                self.stats.update(['URL'])
                continue
            if text in ['…', '...', '..', '.']:
                self.stats.update(['…'])
                continue
            if self.has_forbidden(text):
                self.stats.update(['forbidden'])
                continue
            # Skip prompt bleed-through
            if self.bleed_through(text):
                self.stats.update(['prompt bleed-through'])
                continue
            # Don't repeat yourself
            if f"{self.bot_name}: {text}" in convo:
                self.stats.update(['repetition'])
                continue

            # Too long? Ditch the last sentence fragment.
            if choice['finish_reason'] == 'length':
                try:
                    self.stats.update(['truncated to first sentence'])
                    text = text[:text.rindex('.') + 1]
                except ValueError:
                    pass

            # Now for sentiment analysis. This uses the entire raw response to see where it's leading.
            raw = choice['text'].strip()

            # Potentially on-topic gets a bonus
            nouns_in_reply = [word.lemma_ for word in self.nlp(raw) if word.pos_ == "NOUN"]

            if nouns_in_convo:
                topic_bonus = len(nouns_in_convo.intersection(nouns_in_reply)) / float(len(nouns_in_convo))
            else:
                topic_bonus = 0.0

            all_scores = {
                "flair": get_flair_score(raw),
                "t2e": get_feels_score(raw),
                "profanity": get_profanity_score(raw),
                "topic_bonus": topic_bonus
            }

            # Sum the sentiments, emotional heuristic, offensive quotient, and topic_bonus
            score = sum(all_scores.values()) + topic_bonus
            all_scores['total'] = score
            warning(
                ', '.join([f"{the_score[0]}: {the_score[1]:0.2f}" for the_score in all_scores.items()]),
                "❌" if (score < self.min_score or all_scores['profanity'] < -0.5) else "👍"
            )

            if score < self.min_score:
                self.stats.update(['poor quality'])
                continue

            if all_scores['profanity'] < -0.5:
                self.stats.update(['profanity'])
                continue

            scored[score] = text

        # weights are assumed to be positive. 0 == no chance, so add 1.
        min_score = abs(min(list(scored))) + 1
        adjusted = {}
        for item in scored.items():
            adjusted[item[0] + min_score] = item[1]

        return adjusted

    def get_summary(self, text, summarizer="To sum it up in one sentence:", max_tokens=50):
        ''' Ask GPT for a summary'''
        response = openai.Completion.create(
            engine=self.engine,
            prompt=f"{text}\n\n{summarizer}\n",
            max_tokens=max_tokens,
            top_p=0.1,
            frequency_penalty=0.5,
            presence_penalty=0.0
        )
        reply = response.choices[0]['text'].strip().split('\n')[0]

        # To the right of the : (if any)
        if ':' in reply:
            reply = reply.split(':')[1].strip()

        # Too long? Ditch the last sentence fragment.
        if response.choices[0]['finish_reason'] == "length":
            try:
                reply = reply[:reply.rindex('.') + 1].strip()
            except ValueError:
                pass

        warning("get_summary():", reply)
        return reply

    def has_forbidden(self, text):
        ''' Returns True if any forbidden word appears in text '''
        return bool(re.search(fr'\b({"|".join(self.forbidden)})\b', text))

    def bleed_through(self, text):
        ''' Reject lines that bleed through the standard prompt '''
        for line in (
            "This is a conversation between",
            f"{self.bot_name} is feeling",
            "I am feeling"
        ):
            if line in text:
                return True

        return False

def cleanup(text):
    ''' Strip whitespace and replace \n with space '''
    return text.strip().replace('\n', ' ')
