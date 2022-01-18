''' GPT-3 completion engine '''
import logging
import os
import random
import re

from collections import Counter

import openai
import spacy

from ftfy import fix_text

from feels import get_flair_score, get_feels_score, get_profanity_score

class GPT():
    ''' Container for GPT-3 completion requests '''
    def __init__(
        self,
        bot_name,
        min_score,
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

    def get_best_reply(self, prompt, convo, feels_score=0.0, stop=None, temperature=0.9, max_tokens=200):
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
        logging.debug(f'Prompt: {prompt}\nResponse: {response}')

        # Choose a response based on the most positive sentiment.
        scored = self.score_choices(response.choices, convo)

        if not scored:
            self.stats.update(['replies exhausted'])
            return ':shrug:'

        for item in sorted(scored.items()):
            logging.warning(f"{item[0]:0.2f}: {item[1]}")

        weights = self.calculate_weights(scored, feels_score)

        idx = random.choices(list(sorted(scored)), weights=weights)[0]
        reply = scored[idx]

        logging.warning(f"scores: {sorted(scored)} weights: {weights} choice: {idx} {reply}")
        logging.warning(self.stats)

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
            for sent in sents[1:]:
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
            logging.warning(
                ', '.join([f"{the_score[0]}: {the_score[1]:0.2f}" for the_score in all_scores.items()]) + f' : {raw}'
            )

            if score < self.min_score:
                self.stats.update(['poor quality'])
                continue

            if all_scores['profanity'] < -0.5:
                self.stats.update(['profanity'])
                continue

            scored[score] = text

        return scored

    def calculate_weights(self, scored, feels_score):
        ''' Calculate the weights for scored potential replies '''

        # Start with 1.0 for each reply
        weights = [1.0,] * len(scored)

        # If there's only one, use it
        if len(scored) == 1:
            self.stats.update(['only one reply possible'])
        # If we're feeling too down, take it up a notch
        elif feels_score < 0:
            self.stats.update(['feeling down'])
            weights[0] /= 10
            weights[1] /= 2
        # If we're feeling too good, take it down a notch
        elif feels_score > 0.95:
            self.stats.update(['feeling high'])
            weights[-1] /= 10
        # Otherwise choose randomly
        else:
            self.stats.update(['free choice'])

        return weights

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
            if text.startswith(line):
                return True

        return False

def cleanup(text):
    ''' Strip whitespace and replace \n with space '''
    return text.strip().replace('\n', ' ')
