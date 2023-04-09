''' Generic completion engine wrapper '''
from collections import Counter

from persyn.interaction import gpt  # , nlp_cloud

class LanguageModel():
    ''' Container for language model completion requests '''
    def __init__(
        self,
        config
       ):
        self.config = config
        self.engine = config.completion.engine
        self.bot_name = config.id.name
        self.stats = Counter()

        if not hasattr(config.completion, 'api_key'):
            raise RuntimeError('Please specify completion.api_key in your config file.')

        if self.engine in ['openai', 'gooseai']:
            self.model = gpt.GPT(
                config=config,
            )
            self.max_prompt_length = self.model.max_prompt_length
            self.toklen = self.model.toklen
            self.paginate = self.model.paginate

        # elif self.engine == 'nlpcloud':
        #     model_name = getattr(config.completion, 'model', 'finetuned-gpt-neox-20b')

        #     self.model = nlp_cloud.NLPCLOUD(
        #         config=config
        #     )
        #     self.max_prompt_length = self.model.max_prompt_length
        #     self.toklen = self.model.toklen

        else:
            raise RuntimeError(f'Unknown engine: {self.engine}')

        self.nlp = self.model.nlp
        self.sentiment = self.model.sentiment

    def get_replies(self, prompt, convo, goals=None, stop=None, temperature=0.9, max_tokens=150, n=5, model=None):
        '''
        Given a text prompt and recent conversation, send the prompt to GPT3
        and return a list of possible replies.
        '''
        return self.model.get_replies(prompt, convo, goals, stop, temperature, max_tokens, n, model=model)

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

    def get_summary(self, text, summarizer="To sum it up in one sentence:", max_tokens=50, model=None):
        ''' Ask the model for a summary'''
        return self.model.get_summary(text, summarizer, max_tokens, model)

    def get_keywords(
        self,
        text,
        summarizer="Topics mentioned in the preceding paragraph include the following tags:",
        max_tokens=50
        ):
        ''' Ask the model for keywords'''
        return self.model.get_keywords(text, summarizer, max_tokens)
