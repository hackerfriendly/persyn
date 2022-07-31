''' GPT-3 completion engine '''
import os
import re

from collections import Counter

import openai
import spacy

from ftfy import fix_text

from feels import get_flair_score, get_feels_score, get_profanity_score

# Color logging
from color_logging import ColorLog

log = ColorLog()

class GPT():
    ''' Container for GPT-3 completion requests '''
    def __init__(
        self,
        bot_name,
        min_score=0.0,
        api_key=os.getenv('OPENAI_API_KEY'),
        engine=os.environ.get('OPENAI_MODEL', 'text-davinci-001'),
        forbidden=None,
        max_prompt_length=4000
        ):
        self.bot_name = bot_name
        self.min_score = min_score
        self.engine = engine
        self.forbidden = forbidden or []
        self.max_prompt_length = max_prompt_length
        self.stats = Counter()
        self.nlp = spacy.load("en_core_web_lg")
        openai.api_key = api_key

    def get_replies(self, prompt, convo, stop=None, temperature=0.9, max_tokens=150):
        '''
        Given a text prompt and recent conversation, send the prompt to GPT3
        and return a list of possible replies.
        '''
        if len(prompt) > self.max_prompt_length:
            log.warning(f"get_replies: text too long ({len(prompt)}), truncating to {self.max_prompt_length}")

        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt[:self.max_prompt_length],
            temperature=temperature,
            max_tokens=max_tokens,
            n=8,
            frequency_penalty=1.2,
            presence_penalty=0.8,
            stop=stop
        )
        log.info(f"🧠 Prompt: {prompt}")
        # log.warning(response)

        # Choose a response based on the most positive sentiment.
        scored = self.score_choices(response.choices, convo)
        if not scored:
            self.stats.update(['replies exhausted'])
            log.error("😓 get_replies(): all replies exhausted")
            return None

        log.warning(f"📊 Stats: {self.stats}")

        return scored

    def get_opinions(self, context, entity, stop=[".", "!", "?"], temperature=0.9, max_tokens=50):
        '''
        Ask GPT3 for its opinions of entity, given the context.
        '''
        prompt = f'''{context}\n\nHow does {self.bot_name} feel about {entity}?'''

        if len(prompt) > self.max_prompt_length:
            log.warning(f"get_opinions: prompt too long ({len(prompt)}), truncating to {self.max_prompt_length}")

        response = openai.Completion.create(
            model=self.engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop
        )
        reply = response.choices[0]['text'].strip()
        log.warning(f"☝️ opinion of {entity}: {reply}")

        return reply

    def truncate(self, text):
        '''
        Extract the first few "sentences" from OpenAI's messy output.
        ftfy.fix_text() fixes encoding issues and replaces fancy quotes with ascii.
        spacy parses sentence structure.
        '''
        doc = self.nlp(fix_text(text))
        sents = list(doc.sents)
        if not sents:
            return [':shrug:']
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

        except TypeError:
            pass

        return ' '.join(reply)

    def validate_choice(self, text, convo):
        '''
        Filter low quality GPT responses
        '''
        try:
            # Skip blanks
            if not text:
                self.stats.update(['blank'])
                return None
            # No urls
            if 'http' in text or '.com/' in text:
                self.stats.update(['URL'])
                return None
            if '/r/' in text:
                self.stats.update(['Reddit'])
                return None
            if text in ['…', '...', '..', '.']:
                self.stats.update(['…'])
                return None
            if self.has_forbidden(text):
                self.stats.update(['forbidden'])
                return None
            # Skip prompt bleed-through
            if self.bleed_through(text):
                self.stats.update(['prompt bleed-through'])
                return None
            # Don't repeat yourself for the last three sentences
            if text in ' '.join(convo):
                self.stats.update(['pure repetition'])
                return None
            # Semantic similarity
            choice = self.nlp(text)
            for line in convo:
                if choice.similarity(self.nlp(line)) > 0.97: # TODO: configurable? dynamic?
                    self.stats.update(['semantic repetition'])
                    return None

            return text

        except TypeError:
            log.error(f"🔥 Invalid text for validate_choice(): {text}")
            return None

    def score_choices(self, choices, convo):
        '''
        Filter potential responses for quality, sentimentm and profanity.
        Rank the remaining choices by sentiment and return the ranked list of possible choices.
        '''
        scored = {}

        nouns_in_convo = {word.lemma_ for word in self.nlp(' '.join(convo)) if word.pos_ == "NOUN"}

        for choice in choices:
            text = self.validate_choice(self.truncate(choice['text']), convo)

            if not text:
                continue

            log.debug(f"text: {text}")
            log.debug(f"convo: {convo}")

            # Too long? Ditch the last sentence fragment.
            if choice['finish_reason'] == 'length':
                try:
                    self.stats.update(['truncated to first sentence'])
                    text = text[:text.rindex('.') + 1]
                except ValueError:
                    pass

            # Fix unbalanced symbols
            for symbol in r'(){}[]<>':
                if text.count(symbol) % 2:
                    text = text.replace(symbol, '')
            for symbol in r'"*_':
                if text.count(symbol) % 2:
                    if text.startswith(symbol):
                        text = text + symbol
                    elif text.endswith(symbol):
                        text = symbol + text
                    else:
                        text = text.replace(symbol, '')

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
            log.warning(
                ', '.join([f"{the_score[0]}: {the_score[1]:0.2f}" for the_score in all_scores.items()]),
                "❌" if (score < self.min_score or all_scores['profanity'] < -1.0) else "👍"
            )

            if score < self.min_score:
                self.stats.update(['poor quality'])
                continue

            if all_scores['profanity'] < -1.0:
                self.stats.update(['profanity'])
                continue

            scored[score] = text

        if not scored:
            return {}

        # weights are assumed to be positive. 0 == no chance, so add 1.
        min_score = abs(min(list(scored))) + 1
        adjusted = {}
        for item in scored.items():
            adjusted[item[0] + min_score] = item[1]

        return adjusted

    def get_summary(self, text, summarizer="To sum it up in one sentence:", max_tokens=50):
        ''' Ask GPT for a summary'''
        prompt=f"{text}\n\n{summarizer}\n"
        if len(prompt) > self.max_prompt_length:
            log.warning(f"get_summary: prompt too long ({len(text)}), truncating to {self.max_prompt_length}")
            textlen = self.max_prompt_length - len(summarizer) - 3
            prompt = f"{text[:textlen]}\n\n{summarizer}\n"

        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=max_tokens,
            top_p=0.1,
            frequency_penalty=0.8,
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

        log.warning("gpt get_summary():", reply)
        return reply

    def get_keywords(
        self,
        text,
        summarizer="Topics mentioned in the preceding paragraph include the following tags:",
        max_tokens=50
        ):
        ''' Ask GPT for keywords'''
        keywords = self.get_summary(text, summarizer, max_tokens)
        log.warning(f"gpt get_keywords() raw: {keywords}")

        reply = list(
            {n.text.strip('- ').strip('#').strip('[').strip().lower() for n in self.nlp(keywords).noun_chunks if n.text.strip() != self.bot_name for t in n if t.pos_ != 'PRON'}
        )
        log.warning(f"gpt get_keywords(): {reply}")
        return reply

    def has_forbidden(self, text):
        ''' Returns True if any forbidden word appears in text '''
        if not self.forbidden:
            return False
        return bool(re.search(fr'\b({"|".join(self.forbidden)})\b', text))

    def bleed_through(self, text):
        ''' Reject lines that bleed through the standard prompt '''
        for line in (
            "This is a conversation between",
            f"{self.bot_name} is feeling",
            "I am feeling",
            "I'm feeling",
            f"{self.bot_name}:"
        ):
            if line in text:
                return True

        return False

def cleanup(text):
    ''' Strip whitespace and replace \n with space '''
    return text.strip().replace('\n', ' ')
