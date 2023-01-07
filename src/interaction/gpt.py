''' OpenAI completion engine '''
import re
import string

from collections import Counter

import openai
import spacy

from ftfy import fix_text

from interaction.feels import Sentiment, closest_emoji

# Color logging
from utils.color_logging import ColorLog

log = ColorLog()

class GPT():
    ''' Container for GPT-3 completion requests '''
    def __init__(
        self,
        bot_name,
        min_score,
        api_key,
        api_base,
        model_name,
        forbidden=None,
        nlp=None,
        sentiment=None
        ):
        self.bot_name = bot_name
        self.min_score = min_score
        self.model_name = model_name
        self.forbidden = forbidden or []
        self.stats = Counter()
        self.nlp = nlp or spacy.load("en_core_web_lg")
        self.sentiment = sentiment or Sentiment()

        if model_name.startswith('text-davinci-'):
            self.max_prompt_length = 4000 # tokens
        else:
            self.max_prompt_length = 2048 # tokens

        openai.api_key = api_key
        openai.api_base = api_base

    def get_replies(self, prompt, convo, goals=None, stop=None, temperature=0.9, max_tokens=150):
        '''
        Given a text prompt and recent conversation, send the prompt to GPT3
        and return a list of possible replies.
        '''
        if len(prompt) > self.max_prompt_length:
            log.warning(f"get_replies: text too long ({len(prompt)}), truncating to {self.max_prompt_length}")

        if goals is None:
            goals = []

        try:
            response = openai.Completion.create(
                engine=self.model_name,
                prompt=prompt[:self.max_prompt_length],
                temperature=temperature,
                max_tokens=max_tokens,
                n=8,
                frequency_penalty=1.2,
                presence_penalty=0.8,
                stop=stop
            )
        except openai.error.APIConnectionError as err:
            log.critical("OpenAI APIConnectionError:", err)
            return None
        except openai.error.ServiceUnavailableError as err:
            log.critical("OpenAI Service Unavailable:", err)
            return None
        except openai.error.RateLimitError as err:
            log.critical("OpenAI RateLimitError:", err)
            return None

        log.info(f"🧠 Prompt: {prompt}")
        # log.warning(response)

        # Choose a response based on the most positive sentiment.
        scored = self.score_choices(response.choices, convo, goals)
        if not scored:
            self.stats.update(['replies exhausted'])
            log.error("😓 get_replies(): all replies exhausted")
            return None

        log.warning(f"📊 Stats: {self.stats}")

        return scored

    def get_opinions(self, context, entity, stop=None, temperature=0.9, max_tokens=50):
        '''
        Ask GPT3 for its opinions of entity, given the context.
        '''
        if stop is None:
            stop = [".", "!", "?"]

        prompt = f'''{context}\n\nHow does {self.bot_name} feel about {entity}?'''

        if len(prompt) > self.max_prompt_length:
            log.warning(f"get_opinions: prompt too long ({len(prompt)}), truncating to {self.max_prompt_length}")

        try:
            response = openai.Completion.create(
                engine=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop
            )
        except openai.error.APIConnectionError as err:
            log.critical("OpenAI APIConnectionError:", err)
            return ""
        except openai.error.ServiceUnavailableError as err:
            log.critical("OpenAI Service Unavailable:", err)
            return ""
        except openai.error.RateLimitError as err:
            log.critical("OpenAI RateLimitError:", err)
            return ""

        reply = response.choices[0]['text'].strip()
        log.warning(f"☝️  opinion of {entity}: {reply}")

        return reply

    def get_feels(self, context, stop=None, temperature=0.9, max_tokens=50):
        '''
        Ask GPT3 for sentiment analysis of the current convo.
        '''
        if stop is None:
            stop = [".", "!", "?"]

        prompt = f'''{context}\nThree words that describe {self.bot_name}'s sentiment in the text are:'''

        if len(prompt) > self.max_prompt_length:
            log.warning(f"get_feels: prompt too long ({len(prompt)}), truncating to {self.max_prompt_length}")

        try:
            response = openai.Completion.create(
                engine=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop
            )
        except openai.error.APIConnectionError as err:
            log.critical("OpenAI APIConnectionError:", err)
            return ""
        except openai.error.ServiceUnavailableError as err:
            log.critical("OpenAI Service Unavailable:", err)
            return ""
        except openai.error.RateLimitError as err:
            log.critical("OpenAI RateLimitError:", err)
            return ""

        reply = response.choices[0]['text'].strip()
        log.warning(f"☺️  sentiment of conversation: {reply}")

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
        Filter or fix low quality GPT responses
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
            # No whitespace
            text = text.strip()
            # Putting words Rob: In people's mouths
            match = re.search(r'^(.*)?\s+(\w+: .*)', text)
            if match:
                text = match.group(1)
            # Fix bad emoji
            for match in re.findall(r'(:\S+:)', text):
                closest = closest_emoji(match)
                if match != closest:
                    log.warning(f"😜 {match} > {closest}")
                    text = text.replace(match, closest)
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
            # Don't repeat yourself
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

    def score_choices(self, choices, convo, goals):
        '''
        Filter potential responses for quality, sentimentm and profanity.
        Rank the remaining choices by sentiment and return the ranked list of possible choices.
        '''
        scored = {}

        nouns_in_convo = {word.lemma_ for word in self.nlp(' '.join(convo)) if word.pos_ == "NOUN"}
        nouns_in_goals = {word.lemma_ for word in self.nlp(' '.join(goals)) if word.pos_ == "NOUN"}

        for choice in choices:
            text = self.validate_choice(self.truncate(choice['text']), convo)

            if not text:
                continue

            # if text in choices:
            #     self.stats.update(['pure repetition'])
            #     continue

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

            # Now for sentiment analysis. This uses the entire raw response to see where it's leading.
            raw = choice['text'].strip()

            # Potentially on-topic gets a bonus
            nouns_in_reply = [word.lemma_ for word in self.nlp(raw) if word.pos_ == "NOUN"]

            if nouns_in_convo:
                topic_bonus = len(nouns_in_convo.intersection(nouns_in_reply)) / float(len(nouns_in_convo))
            else:
                topic_bonus = 0.0

            if nouns_in_reply:
                goal_bonus = len(nouns_in_goals.intersection(nouns_in_reply)) / float(len(nouns_in_reply))
            else:
                goal_bonus = 0.0

            all_scores = {
                "flair": self.sentiment.get_sentiment_score(raw),
                "profanity": self.sentiment.get_profanity_score(raw),
                "topic_bonus": topic_bonus,
                "goal_bonus": goal_bonus
            }

            # Sum the sentiments, emotional heuristic, offensive quotient, and topic / goal bonuses
            score = sum(all_scores.values()) + topic_bonus + goal_bonus
            all_scores['total'] = score
            log.warning(
                ', '.join([f"{the_score[0]}: {the_score[1]:0.2f}" for the_score in all_scores.items()]),
                "❌" if (score < self.min_score or all_scores['profanity'] < -1.0) else f"👍 {text}"
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

        try:
            response = openai.Completion.create(
                engine=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                top_p=0.1,
                frequency_penalty=0.8,
                presence_penalty=0.0
            )
        except openai.error.APIConnectionError as err:
            log.critical("OpenAI APIConnectionError:", err)
            return ""
        except openai.error.ServiceUnavailableError as err:
            log.critical("OpenAI Service Unavailable:", err)
            return ""
        except openai.error.RateLimitError as err:
            log.critical("OpenAI RateLimitError:", err)
            return ""

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

    def cleanup_keywords(self, text):
        ''' Tidy up raw completion keywords into a simple list '''
        keywords = []
        bot_name = self.bot_name.lower()

        doc = self.nlp(text)
        for tok in doc:
            keyword = tok.text.strip('#').lstrip('-').strip().lower()
            if keyword != bot_name and keyword not in string.punctuation:
                keywords.append(keyword)

        return sorted(list(set(keywords)))

    def get_keywords(
        self,
        text,
        summarizer="Topics mentioned in the preceding paragraph include the following tags:",
        max_tokens=50
        ):
        ''' Ask GPT for keywords'''
        keywords = self.get_summary(text, summarizer, max_tokens)
        log.warning(f"gpt get_keywords() raw: {keywords}")

        reply = self.cleanup_keywords(keywords)
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
