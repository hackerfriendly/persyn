''' Generic completion engine wrapper '''
import os

from collections import Counter

import spacy

import gpt
import nlp_cloud

class LanguageModel():
    ''' Container for language model completion requests '''
    def __init__(
        self,
        engine,
        bot_name,
        min_score=0.0,
        forbidden=None
       ):
        self.engine = engine
        self.bot_name = bot_name
        self.stats = Counter()
        self.nlp = spacy.load("en_core_web_lg")
        # Absolutely forbidden words
        self.forbidden = forbidden or []
        # Minimum completion reply quality. Lower numbers get more dark + sleazy.
        self.min_score = min_score or float(os.environ.get('MINIMUM_QUALITY_SCORE', -1.0))
        # Temperature. 0.0 == repetitive, 1.0 == chaos
        self.temperature = float(os.environ.get('TEMPERATURE', 0.99))

        if engine in ['openai', 'gooseai']:
            if 'OPENAI_API_KEY' not in os.environ:
                raise RuntimeError('Please specify OPENAI_API_KEY.')
            api_key = os.environ['OPENAI_API_KEY']
            if engine == 'openai':
                api_base = 'https://api.openai.com/v1'
                model_name = os.environ.get('OPENAI_MODEL', 'text-davinci-002')
            else:
                api_base = 'https://api.goose.ai/v1'
                model_name = os.environ.get('OPENAI_MODEL', 'gpt-neo-20b')

            self.model = gpt.GPT(bot_name, self.min_score, api_key, api_base, model_name, forbidden, self.nlp)
            self.max_prompt_length = self.model.max_prompt_length

        elif engine == 'nlpcloud':
            if 'NLPCLOUD_TOKEN' not in os.environ:
                raise RuntimeError('Please specify NLPCLOUD_TOKEN.')
            api_key = os.environ['NLPCLOUD_TOKEN']
            model_name = os.environ.get('NLPCLOUD_MODEL', 'finetuned-gpt-neox-20b')

            self.model = nlp_cloud.NLPCLOUD(bot_name, self.min_score, api_key, model_name, forbidden, self.nlp)
            self.max_prompt_length = self.model.max_prompt_length

        else:
            raise RuntimeError(f'Unknown engine: {engine}')

    def get_replies(self, prompt, convo, goals=None, stop=None, temperature=0.9, max_tokens=150):
        '''
        Given a text prompt and recent conversation, send the prompt to GPT3
        and return a list of possible replies.
        '''
        return self.model.get_replies(prompt, convo, goals, stop, temperature, max_tokens)

    def get_opinions(self, context, entity, stop=None, temperature=0.9, max_tokens=50):
        '''
        Ask the model for its opinions of entity, given the context.
        '''
        if stop is None:
            stop = [".", "!", "?"]
        return self.model.get_opinions(context, entity, stop, temperature, max_tokens)

    def get_feels(self, context, stop=None, temperature=0.9, max_tokens=50):
        '''
        Ask the model for sentiment analysis of the current convo.
        '''
        if stop is None:
            stop = [".", "!", "?"]
        return self.model.get_feels(context, stop, temperature, max_tokens)

    def get_summary(self, text, summarizer="To sum it up in one sentence:", max_tokens=50):
        ''' Ask the model for a summary'''
        return self.model.get_summary(text, summarizer, max_tokens)

    def get_keywords(
        self,
        text,
        summarizer="Topics mentioned in the preceding paragraph include the following tags:",
        max_tokens=50
        ):
        ''' Ask the model for keywords'''
        return self.model.get_keywords(text, summarizer, max_tokens)
