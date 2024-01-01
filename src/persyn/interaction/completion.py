''' LLM completion '''
# pylint: disable=invalid-name

import re

from dataclasses import dataclass
import spacy
import tiktoken

import openai

import numpy as np

from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms.openai import OpenAI, BaseOpenAI
from langchain.globals import set_verbose
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

from ftfy import fix_text

# Color logging
from persyn.utils.color_logging import ColorLog
from persyn.utils.config import PersynConfig

log = ColorLog()

# set_verbose(True)

def setup_llm(config, **kwargs):
    ''' Construct the proper LLM or Chat object for model '''
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

@dataclass
class LanguageModel:
    ''' Container for LLM completion requests '''
    config: PersynConfig

    def __post_init__(self):
        self.bot_name = self.config.id.name
        self.bot_id = self.config.id.guid

        self.chat_model = self.config.completion.chat_model
        self.reasoning_model = self.config.completion.reasoning_model

        self.nlp = spacy.load(self.config.spacy.model)

        openai.api_key = self.config.completion.openai_api_key
        openai.api_base = self.config.completion.openai_api_base
        openai.organization = self.config.completion.openai_org

        self.chat_llm = setup_llm(
            self.config,
            model=self.chat_model,
            temperature=self.config.completion.chat_temperature,
            max_tokens=150,
        )
        self.summary_llm = setup_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.reasoning_temperature,
            max_tokens=100,
        )
        self.feels_llm = setup_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.reasoning_temperature,
            max_tokens=10,
        )

        self.embeddings = OpenAIEmbeddings(openai_api_key=self.config.completion.openai_api_key)

        log.debug(f"💬 chat model: {self.chat_model}")
        log.debug(f"🧠 reasoning model: {self.reasoning_model}")

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

    def trim(self, text):
        ''' Remove junk and any dangling non-sentences from text '''
        # FIXME: Numbered lists are only truncated to the last number, eg. 1. <something>\n2.
        sents = []
        for sent in list(self.nlp(fix_text(text)).sents):
            if sent:
                sents.append(sent)

        if len(sents) > 1 and str(sents[-1][-1]) not in ['.','?','!']:
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

    def get_embedding(self, text, model='text-embedding-ada-002', max_tokens=8192):
        '''
        Return the embedding for text as bytes. Truncates text to the max size supported by the model.
        TODO: embedding model should determine its own size, but embedding models are not (yet?) in BaseOpenAI.modelname_to_contextsize()
        '''
        text = text.replace("\n", " ")

        return  np.array(
            openai.embeddings.create(
                input=[
                    next(
                        self.paginate(text, max_tokens=max_tokens)
                        )
                    ],
                model=model
            ).data[0].embedding,
            dtype=np.float32
        ).tobytes()

    def summarize_text(self, text, summarizer="Summarize the following in one sentence. Your response must include only the summary and no other text:"):
        ''' Ask the LLM for a summary'''
        if not text:
            log.warning('summarize_text():', "No text, skipping summary.")
            return ""

        log.warning(f'summarize_text(): summarizing: {text}')
        prompt = PromptTemplate.from_template(summarizer + "\n{input}")
        chain = prompt | self.summary_llm | StrOutputParser()

        reply = self.trim(chain.invoke({"input": text}))

        # To the right of the Speaker: (if any)
        if re.match(r'^[\w\s]{1,12}:\s', reply):
            reply = reply.split(':')[1].strip()

        log.warning("summarize_text():", reply)
        return reply

    def cosine_similarity(self, a, b):
        ''' Cosine similarity for two embeddings '''
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


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
