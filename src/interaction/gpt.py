''' OpenAI completion engine '''
# pylint: disable=invalid-name

import re

from collections import Counter
from time import sleep

import openai
import spacy
import tiktoken

from ftfy import fix_text

from interaction.feels import Sentiment, closest_emoji

# Color logging
from utils.color_logging import ColorLog

log = ColorLog()

class GPT():
    ''' Container for OpenAI completion requests '''
    def __init__(
        self,
        config
        ):
        self.config = config

        self.forbidden = None
        self.bot_name = config.id.name
        self.bot_id = config.id.guid
        self.min_score = config.completion.minimum_quality_score
        self.completion_model = config.completion.completion_model
        self.chat_model = config.completion.chat_model
        self.summary_model = config.completion.summary_model
        self.nlp = spacy.load(config.spacy.model)

        self.stats = Counter()
        self.sentiment = Sentiment(getattr(config.sentiment, "engine", "flair"),
                                   getattr(config.sentiment, "model", None))

        openai.api_key = config.completion.api_key
        openai.api_base = config.completion.api_base
        openai.organization = config.completion.openai_org

    def get_enc(self, model):
        ''' Return the encoder for model_name '''
        if model is None:
            model = self.completion_model

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding('r50k_base')

    def max_prompt_length(self, model=None):
        ''' Return the maximum number of tokens allowed for a model. '''
        if model is None:
            model = self.completion_model

        if model.startswith('gpt-4'):
            return 8192
        if model.startswith('gpt-3.5') or model.startswith('text-davinci-'):
            return 4097

        # Most models are 2k
        return 2048

    def toklen(self, text, model=None):
        ''' Return the number of tokens in text '''
        return len(self.get_enc(model).encode(text))

    def paginate(self, f, max_tokens=None, prompt=None, max_reply_length=0):
        '''
        Chunk text from iterable f. By default, fit the model's maximum prompt length.
        If prompt is provided, subtract that many tokens from the chunk length.
        Lines containing no alphanumeric characters are removed.
        '''
        if max_tokens is None:
            max_tokens = self.max_prompt_length()

        if prompt:
            max_tokens = max_tokens - self.toklen(prompt)

        max_tokens = max_tokens - max_reply_length

        if isinstance(f, str):
            f = f.split('\n')

        lines = []
        for line in f:
            line = line.strip()
            if not line or not re.search('[a-zA-Z0-9]', line):
                continue

            convo = ' '.join(lines)
            if self.toklen(convo + line) > max_tokens:
                yield convo
                lines = [line]
            else:
                lines.append(line)

        if lines:
            yield ' '.join(lines)

    def get_replies(self, prompt, convo, goals=None, stop=None, temperature=0.9, max_tokens=150, n=5, retry_on_error=True, model=None):
        '''
        Given a text prompt and recent conversation, send the prompt to OpenAI
        and return a list of possible replies.
        '''
        enc = self.get_enc(model)
        if self.toklen(prompt) > self.max_prompt_length():
            log.warning(f"get_replies: text too long ({len(prompt)}), truncating to {self.max_prompt_length()}")
            prompt = enc.decode(enc.encode(prompt)[:self.max_prompt_length()])

        if goals is None:
            goals = []

        if model is None:
            model = self.completion_model

        choices = []
        if model.startswith('gpt-3.5') or model.startswith('gpt-4'):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": """Compose the next line of the following play:"""},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    stop=stop
                )
                choices = [choice['message']['content'] for choice in response['choices']]

            except openai.error.APIConnectionError as err:
                log.critical("get_replies(): OpenAI APIConnectionError:", err)
                return None
            except openai.error.ServiceUnavailableError as err:
                log.critical("get_replies(): OpenAI Service Unavailable:", err)
                return None
            except openai.error.RateLimitError as err:
                log.warning("get_replies(): OpenAI RateLimitError:", err)
                if retry_on_error:
                    log.warning("get_replies(): retrying in 1 second")
                    sleep(1)
                    self.get_replies(prompt, convo, goals, stop, temperature, max_tokens, n=2, retry_on_error=False, model=model)
                return None

        else:
            try:
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    frequency_penalty=1.2,
                    presence_penalty=0.8,
                    stop=stop
                )
                choices = [choice['text'] for choice in response['choices']]

            except openai.error.APIConnectionError as err:
                log.critical("get_replies(): OpenAI APIConnectionError:", err)
                return None
            except openai.error.ServiceUnavailableError as err:
                log.critical("get_replies(): OpenAI Service Unavailable:", err)
                return None
            except openai.error.RateLimitError as err:
                log.warning("get_replies(): OpenAI RateLimitError:", err)
                if retry_on_error:
                    log.warning("get_replies(): retrying in 1 second")
                    sleep(1)
                    self.get_replies(prompt, convo, goals, stop, temperature, max_tokens, n=2, retry_on_error=False)
                return None

        log.info(f"🧠 Prompt: {prompt}")
        log.debug(response)

        # Choose a response based on the most positive sentiment.
        scored = self.score_choices(choices, convo, goals)
        if not scored:
            self.stats.update(['replies exhausted'])
            log.error("😓 get_replies(): all replies exhausted")
            return None

        log.warning(f"📊 Stats: {self.stats}")

        return scored

    def get_opinions(self, context, entity, stop=None, temperature=0.9, max_tokens=50, speaker=None, model=None):
        '''
        Ask ChatGPT for its opinions of entity, given the context.
        '''
        if stop is None:
            stop = [".", "!", "?"]

        if speaker is None:
            speaker = self.bot_name

        if model is None:
            model = self.config.completion.chat_model

        prompt = f'''Given the following conversation, how does {speaker} feel about {entity}?\n{context}'''

        enc = self.get_enc(model)
        if self.toklen(prompt) > self.max_prompt_length():
            log.warning(f"get_opinions: prompt too long ({len(prompt)}), truncating to {self.max_prompt_length()}")
            prompt = enc.decode(enc.encode(prompt)[:self.max_prompt_length()])

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": """You are an expert at estimating opinions based on conversation."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
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

        reply = response['choices'][0]['message']['content'].strip()
        log.warning(f"☝️  opinion of {entity}: {reply}")

        return reply

    def get_feels(self, context, stop=None, temperature=0.9, max_tokens=50, speaker=None, model=None):
        '''
        Ask ChatGPT for sentiment analysis of the current convo.
        '''
        if stop is None:
            stop = [".", "!", "?"]

        if speaker is None:
            speaker = self.bot_name

        if model is None:
            model = self.config.completion.chat_model

        prompt = f"Given the following text, choose three words that best describe {speaker}'s emotional state:\n{context}"

        enc = self.get_enc(model)
        if self.toklen(prompt) > self.max_prompt_length():
            log.warning(f"get_feels: prompt too long ({len(prompt)}), truncating to {self.max_prompt_length()}")
            prompt = enc.decode(enc.encode(prompt)[:self.max_prompt_length()])

        try:
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": """You are an expert at determining the emotional state of people engaging in conversation."""},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
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

        reply = response['choices'][0]['message']['content'].strip().lower()
        log.warning(f"😁 sentiment of conversation: {reply}")

        return reply

    @staticmethod
    def camelCaseName(name):
        ''' Return name sanitized as camelCaseName, alphanumeric only, max 64 characters. '''
        ret = re.sub(r"[^a-zA-Z0-9 ]+", '', name.strip())
        if ' ' in ret:
            words = ret.split(' ')
            ret = ''.join([words[0].lower()] + [w[0].upper()+w[1:].lower() for w in words[1:] if w])
        return ret[:64]

    @staticmethod
    def safe_name(name):
        ''' Return name sanitized as alphanumeric, space, or comma only, max 64 characters. '''
        return re.sub(r"[^a-zA-Z0-9, ]+", '', name.strip())[:64]

    def generate_triples(self, context, temperature=0.5, model=None):
        '''
        Ask ChatGPT to generate a knowledge graph of the current convo.
        Returns a list of (subject, predicate, object) triples.
        '''
        prompt = f"Given the following text, generate a knowledge graph of all important people and facts:\n{context}"

        if model is None:
            model = self.config.completion.chat_model

        enc = self.get_enc(model)
        if self.toklen(prompt) > self.max_prompt_length():
            log.warning(f"get_feels: prompt too long ({len(prompt)}), truncating to {self.max_prompt_length()}")
            prompt = enc.decode(enc.encode(prompt)[:self.max_prompt_length()])

        try:

            response = openai.ChatCompletion.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": """
You are an expert at converting text into knowledge graphs consisting of a subject, predicate, and object separated by | .
The subject, predicate, and object should be as short as possible, consisting of a single word or compoundWord.
Some examples include:
Anna | grewUpIn | Kanata
Anna | hasSibling | Amy
Kanata | locatedNear | Ottawa
Ottawa | locatedIn | Canada
"""
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
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

        reply = response['choices'][0]['message']['content'].strip()

        ret = []
        for line in reply.split('\n'):
            if line.count('|') != 2:
                log.warning('📉 Invalid node:', line)
                continue
            subj, pred, obj = line.split('|')
            subj = self.safe_name(subj)
            pred = self.camelCaseName(pred)
            obj = self.safe_name(obj)
            if not all([subj, pred, obj]):
                continue
            if ',' in obj:
                for o in obj.split(','):
                    safe_obj = self.safe_name(o.strip())
                    if safe_obj:
                        ret.append((subj, pred, safe_obj))
            else:
                ret.append((subj, pred, obj))

        log.info(f"📉 knowledge graph: {len(ret)} triples generated")
        log.debug(f"📉 knowledge graph: {ret}")
        return ret

    def triples_to_text(self, triples, temperature=0.99, preamble=''):
        '''
        Ask ChatGPT to turn a knowledge graph back into text.
        Provide a list of (subject, predicate, object) triples.
        If provided, preamble is inserted in the prompt before graph generation.
        Returns a plain text summary.
        '''
        lines = []
        for triple in triples:
            lines.append(f"{triple[0]} | {triple[1]} | {triple[2]}")

        kg = '\n'.join(lines)
        log.warning(kg)
        try:
            response = openai.ChatCompletion.create(
                model=self.chat_model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": "You are an expert at converting knowledge graphs into succinct text."},
                    {"role": "user", "content":
                    f"""{preamble}
Given the following knowledge graph, create a simple summary of the text it was extracted from
as told from the third-person point of view of {self.bot_name}.

{kg}
"""
                    }
                ]
            )
            text = response['choices'][0]['message']['content'].strip()
            log.info("☘️  triples_to_text:", text)
            return text

        except openai.error.APIConnectionError as err:
            log.critical("OpenAI APIConnectionError:", err)
            return ""
        except openai.error.ServiceUnavailableError as err:
            log.critical("OpenAI Service Unavailable:", err)
            return ""
        except openai.error.RateLimitError as err:
            log.critical("OpenAI RateLimitError:", err)
            return ""

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
        Filter or fix low quality OpenAI responses
        '''
        try:
            # No whitespace or surrounding quotes
            text = text.strip().strip('"')
            # Skip blanks
            if not text:
                self.stats.update(['blank'])
                return None
            # Putting words Rob: In people's mouths
            match = re.search(r'^(.*)?\s+([\w\s]{1,12}: .*)', text)
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
            # Semantic or substring similarity
            choice = self.nlp(text)
            for line in convo:
                if choice.similarity(self.nlp(line)) > 0.97:
                    self.stats.update(['semantic repetition'])
                    return None
                if len(text) > 32 and text.lower() in line.lower():
                    self.stats.update(['simple repetition'])
                    return None

            return text

        except TypeError:
            log.error(f"🔥 Invalid text for validate_choice(): {text}")
            return None

    def score_choices(self, choices, convo, goals):
        '''
        Filter potential responses for quality, sentiment and profanity.
        Rank the remaining choices by sentiment and return the ranked list of possible choices.
        '''
        scored = {}

        nouns_in_convo = {word.lemma_ for word in self.nlp(' '.join(convo)) if word.pos_ == "NOUN"}
        nouns_in_goals = {word.lemma_ for word in self.nlp(' '.join(goals)) if word.pos_ == "NOUN"}

        for choice in choices:
            if not choice:
                continue

            if self.completion_model.startswith('gpt-3.5') or self.completion_model.startswith('gpt-4'):
                text = self.validate_choice(choice, convo)
            else:
                text = self.validate_choice(self.truncate(choice), convo)

            if not text:
                continue

            if re.match(r'^\w+:', text):
                self.stats.update(['putting words in my mouth'])
                continue

            log.debug(f"text: {text}")
            log.debug(f"convo: {convo}")

            # Fix unbalanced symbols
            for symbol in ['()', r'{}', '[]', '<>']:
                if text.count(symbol[0]) != text.count(symbol[1]):
                    text = text.replace(symbol[0], '')
                    text = text.replace(symbol[1], '')

            # Now for sentiment analysis. This uses the entire raw response to see where it's leading.

            # Potentially on-topic gets a bonus
            nouns_in_reply = [word.lemma_ for word in self.nlp(choice) if word.pos_ == "NOUN"]

            if nouns_in_convo:
                topic_bonus = len(nouns_in_convo.intersection(nouns_in_reply)) / float(len(nouns_in_convo))
            else:
                topic_bonus = 0.0

            if nouns_in_reply:
                goal_bonus = len(nouns_in_goals.intersection(nouns_in_reply)) / float(len(nouns_in_reply))
            else:
                goal_bonus = 0.0

            all_scores = {
                "flair": self.sentiment.get_sentiment_score(choice),
                "profanity": self.sentiment.get_profanity_score(choice),
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

    def get_summary(self, text, summarizer="To sum it up in one sentence:", max_tokens=50, model=None):
        ''' Ask ChatGPT for a summary'''
        if not text:
            log.warning('get_summary():', "No text, skipping summary.")
            return ""

        prompt=f"{text}\n\n{summarizer}\n"

        if model is None:
            model = self.config.completion.summary_model

        enc = self.get_enc(model)
        if self.toklen(prompt) > self.max_prompt_length(model):
            log.warning(f"get_summary: prompt too long ({len(text)}), truncating to {self.max_prompt_length(model)}")
            prompt = enc.decode(enc.encode(prompt)[:self.max_prompt_length(model)])

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": """You are an expert at summarizing text."""},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                top_p=0.1,
                frequency_penalty=0.8,
                presence_penalty=0.0,
                n=1
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

        reply = response['choices'][0]['message']['content'].strip() #.split('\n')[0]

        # To the right of the Speaker: (if any)
        if re.match(r'^[\w\s]{1,12}:\s', reply):
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

        for kw in [item.strip() for line in text.split('\n') for item in line.split(',')]:
            # Regex chosen by GPT-4 to match bulleted lists (#*-) or numbered lists. 😵‍💫
            match = re.search(r'^\s*(?:\d+\.\s+|\*\s+|-{1}\s+|#\s*)?(.*)', kw)
            # At least one alpha required
            if match and re.match(r'.*[a-zA-Z]', match.group(1)):
                kw = match.group(1).strip()
            else:
                kw = kw.strip()

            if kw.lower() != bot_name:
                keywords.append(kw)

        return sorted(set(keywords))

    def get_keywords(
        self,
        text,
        summarizer="Topics mentioned in the preceding paragraph include the following tags:",
        max_tokens=50
        ):
        ''' Ask OpenAI for keywords'''
        keywords = self.get_summary(text, summarizer, max_tokens)
        log.debug(f"gpt get_keywords() raw: {keywords}")

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
