''' OpenAI completion '''
# pylint: disable=invalid-name

import re

from collections import Counter
from typing import List, Optional

import spacy
import tiktoken

import openai

import numpy as np

from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI, BaseOpenAI
from langchain.chains import LLMChain

from ftfy import fix_text

# Color logging
from persyn.utils.color_logging import ColorLog
from persyn.interaction.feels import closest_emoji

log = ColorLog()

def get_oai_embedding(text: str, model="text-embedding-ada-002", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.embeddings.create(input=[text], model=model, **kwargs).data[0].embedding

def the_llm(**kwargs):
    ''' Construct the proper LLM object for model '''
    if kwargs['model'].startswith('gpt-'):
        return ChatOpenAI(**kwargs)
    return OpenAI(**kwargs)

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

        openai.api_key = config.completion.api_key
        openai.api_base = config.completion.api_base
        openai.organization = config.completion.openai_org

        self.completion_llm = the_llm(
            model=self.completion_model,
            temperature=self.config.completion.temperature,
            max_tokens=150,
            openai_api_key=self.config.completion.api_key,
            openai_organization=self.config.completion.openai_org
        )
        self.summary_llm = the_llm(
            model=self.summary_model,
            temperature=self.config.completion.temperature,
            max_tokens=50,
            openai_api_key=self.config.completion.api_key,
            openai_organization=self.config.completion.openai_org
        )
        self.feels_llm = the_llm(
            model=self.completion_model,
            temperature=self.config.completion.temperature,
            max_tokens=10,
            openai_api_key=self.config.completion.api_key,
            openai_organization=self.config.completion.openai_org,
        )

        log.debug(f"🤖 completion model: {self.completion_model}")
        log.debug(f"🤖 summary model: {self.summary_model}")

    def get_enc(self, model=None):
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

        try:
            return BaseOpenAI.modelname_to_contextsize(model)
        except ValueError as err:
            if model == 'gpt-4-1106-preview':
                return 128 * 1024
            else:
                raise err

    def toklen(self, text, model=None):
        ''' Return the number of tokens in text '''
        if model is None:
            model = self.completion_model
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

    def validate_reply(self, text: str):
        '''
        Filter or fix low quality OpenAI responses
        '''
        try:
            # No whitespace or surrounding quotes
            text = str(text).strip().strip('"\'')
            # Skip blanks
            if not text:
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
                return None
            if text in ['…', '...', '..', '.']:
                return None
            if self.has_forbidden(text):
                return None
            # Skip prompt bleed-through
            if self.bleed_through(text):
                return None

            return text

        except TypeError:
            log.error(f"🔥 Invalid text for validate_choice(): {text}")
            return None

    def trim(self, text):
        ''' Remove junk and any dangling non-sentences from text '''
        sents = []
        for sent in list(self.nlp(fix_text(text)).sents):
            poss = self.validate_reply(sent)
            if poss:
                sents.append(self.nlp(poss))

        if len(sents) > 1 and not sents[-1][-1].is_punct:
            sents.pop()

        return ' '.join([sent.text for sent in sents])

    def truncate(self, text, model=None):
        ''' Truncate text to the max_prompt_length for this model '''
        if model is None:
            model = self.completion_model

        maxlen = self.max_prompt_length(model)
        if self.toklen(text) <= maxlen:
            return text

        log.warning(f"truncate: text too long ({self.toklen(text)}), truncating to {maxlen}")
        enc = self.get_enc(model)
        return enc.decode(enc.encode(text)[:maxlen])

    def get_embedding(self, text, model='text-embedding-ada-002'):
        ''' Return the embedding for text '''
        return  np.array(get_oai_embedding(text, model=model), dtype=np.float32).tobytes()

    def cosine_similarity(self, a, b):
        ''' Cosine similarity for two embeddings '''
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_reply(self, prompt):
        '''
        Send the prompt to the LLM and return the top reply.
        '''
        prompt = self.truncate(prompt)

        template = """Compose the next line of the following play:\n{prompt}"""
        llm_chain = LLMChain.from_string(llm=self.completion_llm, template=template)
        # response = self.trim(llm_chain.predict(prompt=prompt))
        response = llm_chain.predict(prompt=prompt)

        if not response:
            log.warning("🤔 No reply, trying again...")
            response = llm_chain.predict(prompt=prompt)

        log.info(f"🧠 Prompt: {prompt}")
        log.info(f"🧠 👉 {response}")

        return response

    def get_opinions(self, context, entity):
        '''
        Ask the LLM for its opinions of entity, given the context.
        '''
        if model is None:
            model = self.config.completion.chat_model

        log.warning("🧷 get_opinions:", entity)
        prompt = self.truncate(
            f"Briefly state {self.bot_name}'s opinion about {entity} from {self.bot_name}'s point of view, and convert pronouns and verbs to the first person.\n{context}",
            model=self.summary_model
        )

        template = """You are an expert at estimating opinions based on conversation.\n{prompt}"""
        llm_chain = LLMChain.from_string(llm=self.summary_llm, template=template)
        reply = self.trim(llm_chain.predict(prompt=prompt).strip())

        log.warning(f"☝️  opinion of {entity}: {reply}")

        return reply

    def get_feels(self, context):
        '''
        Ask the LLM for sentiment analysis of the current convo.
        '''
        prompt = self.truncate(
            f"In the following text, these three comma separated words best describe {self.bot_name}'s emotional state:\n{context}",
            model=self.completion_model
        )

        template = """You are an expert at determining the emotional state of people engaging in conversation.\n{prompt}"""
        llm_chain = LLMChain.from_string(llm=self.feels_llm, template=template)

        reply = self.trim(llm_chain.predict(prompt=prompt).strip().lower())

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
        Ask the LLM to generate a knowledge graph of the current convo.
        Returns a list of (subject, predicate, object) triples.
        '''
        prompt = self.truncate(
            f"Given the following text, generate a knowledge graph of all important people and facts:\n{context}"
        )

        if model is None:
            model = self.config.completion.chat_model

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
        Ask the LLM to turn a knowledge graph back into text.
        Provide a list of (subject, predicate, object) triples.
        If provided, preamble is inserted in the prompt before graph generation.
        Returns a plain text summary.
        '''
        lines = []
        for triple in triples:
            lines.append(f"{triple[0]} | {triple[1]} | {triple[2]}")

        log.info(f"☘️  {len(lines)} triples to summarize")
        kg = '\n'.join(lines)
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
            log.info("☘️  graph summary:", text)
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

    def get_summary(self, text, summarizer="Sum the following up in one sentence:"):
        ''' Ask the LLM for a summary'''
        if not text:
            log.warning('get_summary():', "No text, skipping summary.")
            return ""

        prompt=self.truncate(f"{summarizer}\n\n{text}", model=self.config.completion.summary_model)

        template = """You are an expert at summarizing text.\n{prompt}"""
        llm_chain = LLMChain.from_string(llm=self.summary_llm, template=template)
        reply = self.trim(llm_chain.predict(prompt=prompt).strip())

        # To the right of the Speaker: (if any)
        if re.match(r'^[\w\s]{1,12}:\s', reply):
            reply = reply.split(':')[1].strip()

        log.warning("gpt get_summary():", reply)
        return reply

    def cleanup_keywords(self, text):
        ''' Tidy up raw completion keywords into a simple list '''
        keywords = []
        bot_name = self.bot_name.lower()

        for kw in [item.strip() for line in text.replace('#', '\n').split('\n') for item in line.split(',')]:
            # Regex chosen by GPT-4 to match bulleted lists (#*-) or numbered lists, with further tweaks. 😵‍💫
            match = re.search(r'^\s*(?:\d+\.\s+|\*\s+|-{1}\s*|#\s*)?(.*)', kw)
            # At least one alpha required
            if match and re.match(r'.*[a-zA-Z]', match.group(1)):
                kw = match.group(1).strip()
            elif re.match(r'.*[a-zA-Z]', kw):
                kw = kw.strip()
            else:
                continue

            if kw.lower() != bot_name:
                keywords.append(kw.lower())

        return sorted(set(keywords))

    def get_keywords(
        self,
        text,
        summarizer="Topics mentioned in the preceding paragraph include the following tags:"
        ):
        ''' Ask OpenAI for keywords'''
        keywords = self.get_summary(text, summarizer)
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
            f"{self.bot_name} feels:",
            "I am feeling",
            "I'm feeling",
            f"{self.bot_name}:"
        ):
            if text.startswith(line):
                return True

        return False

def cleanup(text):
    ''' Strip whitespace and replace \n with space '''
    return text.strip().replace('\n', ' ')
