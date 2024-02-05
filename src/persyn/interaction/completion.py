''' LLM completion '''
# pylint: disable=invalid-name

import json
import re

from dataclasses import dataclass
from time import sleep
from typing import Optional, Union, Set
import pydantic
import spacy
import tiktoken

import openai
import anthropic

import numpy as np

from ftfy import fix_text

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.llms.base import BaseOpenAI
from langchain_community.chat_models import ChatAnthropic

# from langchain.globals import set_verbose
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

# Color logging
from persyn.utils.color_logging import ColorLog
from persyn.utils.config import PersynConfig

log = ColorLog()

# set_verbose(True)

def setup_llm(config, **kwargs) -> Union[ChatOpenAI, ChatAnthropic, None]:
    ''' Construct the proper Chat object for model. TODO: support for other LLMs. '''
    if kwargs['model'].startswith('claude-'):
        try:
            return ChatAnthropic(
                anthropic_api_key=config.completion.anthropic_key,
                **kwargs
            )
        except pydantic.ValidationError:
            log.warning('anthropic_key not found in config.completion, skipping Anthropic support.')
            return None

    return ChatOpenAI(
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
            max_tokens=200,
        )
        self.summary_llm = setup_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.reasoning_temperature,
            max_tokens=100,
        )
        self.final_summary_llm = setup_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.reasoning_temperature,
            max_tokens=300,
        )
        self.feels_llm = setup_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.reasoning_temperature,
            max_tokens=10,
        )
        self.anthropic_llm = setup_llm(
            self.config,
            model=self.config.completion.anthropic_model,
            temperature=self.config.completion.anthropic_temperature,
            max_tokens=250,
        )
        self.reflection_llm = setup_llm(
            self.config,
            model=self.reasoning_model,
            temperature=self.config.completion.reasoning_temperature,
            max_tokens=500,
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.completion.openai_api_key,
            model=self.config.memory.embedding_model
        )

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

            if total + tl > max_tokens:
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
        sents = []
        for sent in list(self.nlp(fix_text(text)).sents):
            if sent:
                sents.append(sent)

        if len(sents) > 1 and str(sents[-1][-1]) not in ['.','?','!']:
            sents.pop()

        if len(sents) > 1 and len(sents[-1]) < 4:
            sents.pop()

        return str(' '.join([sent.text for sent in sents])).strip()

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

    def get_embedding(self, text, model=None, max_tokens=8192):
        '''
        Return the embedding for text as bytes. Truncates text to the max size supported by the model.
        TODO: embedding model should determine its own size, but embedding models are not (yet?) in BaseOpenAI.modelname_to_contextsize()
        '''
        if model is None:
            model = self.config.memory.embedding_model

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

    def summarize_text(
        self,
        text: str,
        summarizer: str = "Summarize the following in one sentence. Your response must include only the summary and no other text:",
        final: Optional[bool] = False) -> str:
        ''' Ask the LLM for a summary. If final is True, use the final_summary_llm. '''

        if final:
            llm = self.final_summary_llm
        else:
            llm = self.summary_llm

        text = self.truncate(text.strip()) # FIXME: proper model selection here
        if not text:
            log.warning('summarize_text():', "No text, skipping summary.")
            return ""

        log.warning(f'summarize_text(): summarizing: {text[:100]}…')

        prompt = PromptTemplate.from_template(summarizer + "\n{input}")
        chain = prompt | llm | StrOutputParser()

        reply = self.trim(chain.invoke({"input": text}))

        # To the right of the Speaker: (if any)
        if re.match(r'^[\w\s]{1,12}:\s', reply):
            reply = reply.split(':')[1].strip()

        log.warning("summarize_text():", reply)
        return reply

    def reflect(
        self,
        text: str
    ) -> Union[dict[str, list[str]], None]:
        ''' Ask the LLM to reflect on text. Returns a dict of {question: [answer, answer, answer]} '''

        llm = self.reflection_llm

        text = self.truncate(text) # FIXME: proper model selection here
        if not text:
            log.warning('reflect():', "No text, skipping.")
            return None

        prompt = PromptTemplate.from_template(
            f"""Given only following dialog, list up to two salient high-level questions that can be asked about {self.config.id.name}'s goals, desires, and opinions.
For each question, also list up to two specific actions that {self.config.id.name} can take to answer those questions.
Make your answers as concise as possible. Convert pronouns and verbs to the first person, and format your reply using JSON in the following format:
""" + """
{{
    "First question": [ "Answers to question 1", "as a list" ],
    "Second optional question": [ "Answers to question 2", "as a list" ]
}}

Your response MUST only include JSON, no other text or preamble. Your response MUST return valid JSON, with no ``` or other special formatting.\n
{input}""")

        chain = prompt | llm | StrOutputParser()

        reply = chain.invoke({"input": text})

        # Try to remove any preamble and ``` if present
        if not reply.startswith('{'):
            reply = '{' + reply.split('{', 1)[1].replace('`', '')

        try:
            ret = json.loads(reply)
        except json.decoder.JSONDecodeError as err:
            log.error("reflect(): Could not parse JSON response:", str(err))
            return None

        return ret

    def cosine_similarity(self, a, b):
        ''' Cosine similarity for two embeddings '''
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
                keywords.append(kw)

        return sorted(set(keywords))

    def ask_claude(self, query: str, prefix: Optional[str] = '') -> str:
        ''' Ask Claude a question '''
        if not query:
            log.warning('ask_claude():', "No query, skipping.")
            return ""

        prompt = PromptTemplate.from_template(f"{prefix}{{input}}")

        try:
            chain = prompt | self.anthropic_llm | StrOutputParser()
        except anthropic.RateLimitError:
            log.warning('ask_claude():', "Rate limit error, retrying.")
            sleep(2)
            chain = prompt | self.anthropic_llm | StrOutputParser()

        reply = self.trim(chain.invoke({"input": query}))

        log.warning("ask_claude():", reply)
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

    def extract_nouns(self, text: str) -> list[str]:
        ''' return a list of all nouns (except pronouns) in text '''
        doc = self.nlp(text)
        nouns = {
            n.text.strip()
            for n in doc.noun_chunks
            if n.text.strip() != self.config.id.name
            for t in n
            if t.pos_ != 'PRON'
        }
        return list(nouns)

    def extract_entities(self, text: str) -> Set[str]:
        ''' Return a set of all entities in text '''
        doc = self.nlp(text)
        return {n.text.strip() for n in doc.ents if len(n.text.strip()) > 2}
