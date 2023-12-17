''' LLM completion '''
# pylint: disable=invalid-name

import re

from collections import Counter
from typing import List, Optional

import spacy
import tiktoken

import openai

import numpy as np

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms.openai import OpenAI, BaseOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

from ftfy import fix_text

from langchain.globals import set_verbose

# Color logging
from persyn.utils.color_logging import ColorLog
from persyn.interaction.feels import closest_emoji

log = ColorLog()

set_verbose(True)

def get_oai_embedding(text: str, model="text-embedding-ada-002", **kwargs) -> List[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.embeddings.create(input=[text], model=model, **kwargs).data[0].embedding

def the_llm(config, **kwargs):
    ''' Construct the proper LLM object for model '''
    if kwargs['model'].startswith('gpt-'):
        return ChatOpenAI(
            openai_api_key=config.completion.openai_api_key,
            openai_organization=config.completion.openai_org,
            **kwargs
        )
    if kwargs['model'].startswith('claude-'):
        return ChatAnthropic(
            anthropic_api_key=config.completion.anthropic_key,
            **kwargs
        )
    return OpenAI(
        openai_api_key=config.completion.openai_api_key,
        openai_organization=config.completion.openai_org,
        **kwargs
    )

class LanguageModel():
    ''' Container for OpenAI completion requests '''
    def __init__(
        self,
        config
        ):
        self.config = config

        self.forbidden = None
        self.bot_name = config.id.name
        self.bot_id = config.id.guid

        self.chat_model = config.completion.chat_model
        self.reasoning_model = config.completion.reasoning_model

        self.nlp = spacy.load(config.spacy.model)

        self.stats = Counter()

        openai.api_key = config.completion.openai_api_key
        openai.api_base = config.completion.openai_api_base
        openai.organization = config.completion.openai_org

        self.completion_llm = the_llm(
            self.config,
            model=self.chat_model,
            temperature=self.config.completion.chat_temperature,
            max_tokens=150,
        )
        self.summary_llm = the_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.reasoning_temperature,
            max_tokens=100,
        )
        self.feels_llm = the_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.chat_temperature,
            max_tokens=10,
        )

        log.debug(f"🤖 chat model: {self.chat_model}")
        log.debug(f"🤖 reasoning model: {self.reasoning_model}")

    def get_enc(self, model=None):
        ''' Return the encoder for model_name '''
        if model is None:
            model = self.chat_model

        try:
            return tiktoken.encoding_for_model(model)
        except KeyError:
            return tiktoken.get_encoding('r50k_base')

    def max_prompt_length(self, model=None):
        ''' Return the maximum number of tokens allowed for a model. '''
        if model is None:
            model = self.chat_model

        try:
            return BaseOpenAI.modelname_to_contextsize(model)
        except ValueError as err:
            # Anthropic
            if model.startswith('claude-'):
                if model == 'claude-2.1':
                    return 200000
                return 100000
            # Beta OpenAI models
            if model == 'gpt-4-1106-preview':
                return 128 * 1024
            # Unknown model
            raise err

    def toklen(self, text, model=None):
        ''' Return the number of tokens in text '''
        if model is None:
            model = self.chat_model
        return len(self.get_enc(model).encode(text))

    def paginate(self, f, max_tokens=None, prompt=None, max_reply_length=0):
        '''
        Chunk text from iterable f, splitting on whitespace, chunk by tokens.
        By default, fit the model's maximum prompt length.
        If prompt is provided, subtract that many tokens from the chunk length.
        '''
        if max_tokens is None:
            max_tokens = self.max_prompt_length()

        # 1 token minimum
        max_tokens = max(1, max_tokens)

        if prompt:
            max_tokens = max_tokens - self.toklen(prompt)

        max_tokens = max_tokens - max_reply_length

        if isinstance(f, str):
            f = f.split()

        words = []
        total = 0
        for word in f:
            tl = self.toklen(word)
            if not tl:
                continue

            if total + tl >= max_tokens:
                ret = ' '.join(words)
                total = tl
                words = [word]
                yield ret
            else:
                total = total + tl
                words.append(word)

        if words:
            yield ' '.join(words)

    def fixup_reply(self, text: str):
        '''
        Filter or fix low quality responses
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
            # Claude often includes *stage directions* despite being asked not to
            text = re.sub(r'[*].*?[*] +?', '', text)

            return text

        except TypeError:
            log.error(f"🔥 Invalid text for validate_choice(): {text}")
            return None

    def trim(self, text):
        ''' Remove junk and any dangling non-sentences from text '''
        sents = []
        for sent in list(self.nlp(fix_text(text)).sents):
            poss = self.fixup_reply(sent)
            if poss:
                sents.append(self.nlp(poss))

        if len(sents) > 1 and not sents[-1][-1].is_punct:
            sents.pop()

        return ' '.join([sent.text for sent in sents])

    def truncate(self, text, model=None):
        ''' Truncate text to the max_prompt_length for this model '''
        if model is None:
            model = self.chat_model

        maxlen = self.max_prompt_length(model)
        if self.toklen(text) <= maxlen:
            return text

        log.warning(f"truncate: text too long ({self.toklen(text)}), truncating to {maxlen}")
        enc = self.get_enc(model)
        return enc.decode(enc.encode(text)[:maxlen])

    def get_embedding(self, text, model='text-embedding-ada-002'):
        ''' Return the embedding for text. Truncates text to the max size supported by the model. '''
        # TODO: embedding model should determine its own size, but embedding models are not (yet?) in BaseOpenAI.modelname_to_contextsize()

        return  np.array(
            get_oai_embedding(
                next(self.paginate(text, max_tokens=8192)),
                model=model
            ),
            dtype=np.float32
        ).tobytes()

    def cosine_similarity(self, a, b):
        ''' Cosine similarity for two embeddings '''
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_reply(self, prompt):
        '''
        Send the prompt to the LLM.
        '''
        prompt = self.truncate(prompt)

        ex = ChatPromptTemplate.from_messages([
            ("human", "I will present you with a fictional dialog. Please respond by continuing the dialog from {bot_name}'s point of view, always responding in the first person. I understand if you cannot provide an emotional perspective, but you can use sentiment analysis of the text instead. If you don't have enough context, do the best you can with what is provided and do not break character under ANY circumstances. You must provide only the next line of the dialog. Do you understand?"),
            ("ai", "Yes, I understand the instructions. I will continue the dialog to the best of my ability."),
            ("human", prompt)
        ])

        chain = ex | self.completion_llm
        response = chain.invoke({'bot_name': self.config.id.name}).content

        response = self.trim(response) or response

        if not response:
            log.warning("🤔 No reply, trying again...")
            response = chain.invoke({'bot_name': self.config.id.name}).content

        log.info(f"🧠 Prompt: {prompt}")
        # log.info(f"🧠 Converted: {self.completion_llm.convert_prompt(ex.format_prompt(bot_name=self.config.id.name))}")
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
            model=self.reasoning_model
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
            model=self.chat_model
        )

        template = """
You are an expert at determining the emotional state of people engaging in conversation.
{prompt}
-----
Your response should only include the three words, no other text.
"""
        llm_chain = LLMChain.from_string(llm=self.feels_llm, template=template)

        reply = self.trim(llm_chain.predict(prompt=prompt).strip().lower())

        log.warning(f"😁 sentiment of conversation: {reply}")

        return reply

    def fact_check(self, context):
        '''
        Ask the LLM to fact check the current convo.
        '''
        log.debug(f"✅ fact check: {context}")

        prompt = self.truncate(
            f"Examine all facts in the following conversation, pointing out any inconsistencies. Convert pronouns and verbs to the first person:\n{context}",
            model=self.reasoning_model
        )

        template = """
You are an experienced fact-checker, and are happy to validate any inconsistencies in a dialog.
{prompt}
"""
        llm_chain = LLMChain.from_string(llm=self.summary_llm, template=template)

        reply = self.trim(llm_chain.predict(prompt=prompt).strip())

        log.warning(f"✅ fact check: {reply}")

        if 'NONE' in reply:
            return None

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
        pass
#         '''
#         Ask the LLM to generate a knowledge graph of the current convo.
#         Returns a list of (subject, predicate, object) triples.
#         '''
#         prompt = self.truncate(
#             f"Given the following text, generate a knowledge graph of all important people and facts:\n{context}"
#         )

#         if model is None:
#             model = self.config.completion.chat_model

#         try:
#             response = openai.ChatCompletion.create(
#                 model=self.chat_model,
#                 messages=[
#                     {"role": "system", "content": """
# You are an expert at converting text into knowledge graphs consisting of a subject, predicate, and object separated by | .
# The subject, predicate, and object should be as short as possible, consisting of a single word or compoundWord.
# Some examples include:
# Anna | grewUpIn | Kanata
# Anna | hasSibling | Amy
# Kanata | locatedNear | Ottawa
# Ottawa | locatedIn | Canada
# """
#                     },
#                     {"role": "user", "content": prompt}
#                 ],
#                 temperature=temperature
#             )
#         except openai.error.APIConnectionError as err:
#             log.critical("OpenAI APIConnectionError:", err)
#             return ""
#         except openai.error.ServiceUnavailableError as err:
#             log.critical("OpenAI Service Unavailable:", err)
#             return ""
#         except openai.error.RateLimitError as err:
#             log.critical("OpenAI RateLimitError:", err)
#             return ""

#         reply = response['choices'][0]['message']['content'].strip()

#         ret = []
#         for line in reply.split('\n'):
#             if line.count('|') != 2:
#                 log.warning('📉 Invalid node:', line)
#                 continue
#             subj, pred, obj = line.split('|')
#             subj = self.safe_name(subj)
#             pred = self.camelCaseName(pred)
#             obj = self.safe_name(obj)
#             if not all([subj, pred, obj]):
#                 continue
#             if ',' in obj:
#                 for o in obj.split(','):
#                     safe_obj = self.safe_name(o.strip())
#                     if safe_obj:
#                         ret.append((subj, pred, safe_obj))
#             else:
#                 ret.append((subj, pred, obj))

#         log.info(f"📉 knowledge graph: {len(ret)} triples generated")
#         log.debug(f"📉 knowledge graph: {ret}")
#         return ret

    def triples_to_text(self, triples, temperature=0.99, preamble=''):
        pass
#         '''
#         Ask the LLM to turn a knowledge graph back into text.
#         Provide a list of (subject, predicate, object) triples.
#         If provided, preamble is inserted in the prompt before graph generation.
#         Returns a plain text summary.
#         '''
#         lines = []
#         for triple in triples:
#             lines.append(f"{triple[0]} | {triple[1]} | {triple[2]}")

#         log.info(f"☘️  {len(lines)} triples to summarize")
#         kg = '\n'.join(lines)
#         try:
#             response = openai.ChatCompletion.create(
#                 model=self.chat_model,
#                 temperature=temperature,
#                 messages=[
#                     {"role": "system", "content": "You are an expert at converting knowledge graphs into succinct text."},
#                     {"role": "user", "content":
#                     f"""{preamble}
# Given the following knowledge graph, create a simple summary of the text it was extracted from
# as told from the third-person point of view of {self.bot_name}.

# {kg}
# """
#                     }
#                 ]
#             )
#             text = response['choices'][0]['message']['content'].strip()
#             log.info("☘️  graph summary:", text)
#             return text

#         except openai.error.APIConnectionError as err:
#             log.critical("OpenAI APIConnectionError:", err)
#             return ""
#         except openai.error.ServiceUnavailableError as err:
#             log.critical("OpenAI Service Unavailable:", err)
#             return ""
#         except openai.error.RateLimitError as err:
#             log.critical("OpenAI RateLimitError:", err)
#             return ""

    def get_summary(self, text, summarizer="Summarize the following in one sentence. Your response must include only the summary and no other text."):
        ''' Ask the LLM for a summary'''
        if not text:
            log.warning('get_summary():', "No text, skipping summary.")
            return ""

        prompt=self.truncate(f"{summarizer}\n\nHuman:\n{text}\n-----\nAssistant: ", model=self.config.completion.reasoning_model)
        log.warning(f'get_summary(): summarizing: {prompt}')
        template = "{prompt}"
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
        ''' Ask for keywords'''
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
