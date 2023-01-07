''' Generic completion engine wrapper '''
from collections import Counter

import spacy

from interaction import gpt, nlp_cloud
from interaction.feels import Sentiment

class LanguageModel():
    ''' Container for language model completion requests '''
    def __init__(
        self,
        config,
        forbidden=None
       ):
        self.config = config
        self.engine = config.completion.engine
        self.bot_name = config.id.name
        self.stats = Counter()
        self.nlp = spacy.load(getattr(config.completion, "spacy_model", "en_core_web_lg"))
        self.sentiment = Sentiment(getattr(config.sentiment, "engine", "flair"),
                                   getattr(config.sentiment, "model", None))
        # Absolutely forbidden words
        self.forbidden = forbidden or []
        # Minimum completion reply quality. Lower numbers get more dark + sleazy.
        self.min_score = float(getattr(config.completion, 'minimum_quality_score', -1.0))
        # Temperature. 0.0 == repetitive, 1.0 == chaos
        self.temperature = float(getattr(config.completion, 'temperature', 0.99))

        if not hasattr(config.completion, 'api_key'):
            raise RuntimeError('Please specify completion.api_key in your config file.')

        if self.engine in ['openai', 'gooseai']:
            if self.engine == 'openai':
                api_base = 'https://api.openai.com/v1'
                model_name = getattr(config.completion, 'model', 'text-davinci-003')
            else:
                api_base = 'https://api.goose.ai/v1'
                model_name = getattr(config.completion, 'model', 'gpt-neo-20b')

            self.model = gpt.GPT(
                self.bot_name,
                self.min_score,
                config.completion.api_key,
                api_base,
                model_name,
                forbidden,
                self.nlp
            )
            self.max_prompt_length = self.model.max_prompt_length

        elif self.engine == 'nlpcloud':
            model_name = getattr(config.completion, 'model', 'finetuned-gpt-neox-20b')

            self.model = nlp_cloud.NLPCLOUD(
                self.bot_name,
                self.min_score,
                config.completion.api_key,
                model_name,
                forbidden,
                self.nlp
            )
            self.max_prompt_length = self.model.max_prompt_length

        else:
            raise RuntimeError(f'Unknown engine: {self.engine}')

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
