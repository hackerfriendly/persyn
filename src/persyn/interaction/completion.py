''' Generic completion engine wrapper '''
from collections import Counter

from persyn.interaction import gpt

class LanguageModel():
    ''' Container for language model completion requests '''
    def __init__(
        self,
        config
       ):
        self.config = config
        self.bot_name = config.id.name
        self.stats = Counter()

        self.model = gpt.GPT(
            config=config,
        )
        self.max_prompt_length = self.model.max_prompt_length
        self.toklen = self.model.toklen
        self.paginate = self.model.paginate

        self.nlp = self.model.nlp

    def get_reply(self, prompt):
        '''
        Send the prompt to the LLM and return the top reply.
        '''
        return self.model.get_reply(prompt)

    def get_opinions(self, context, entity):
        '''
        Ask the model for its opinions of entity, given the context.
        '''
        return self.model.get_opinions(context, entity)

    def get_feels(self, context):
        '''
        Ask the model for sentiment analysis of the current convo.
        '''
        return self.model.get_feels(context)

    def get_summary(self, text, summarizer="To sum it up in one sentence:"):
        ''' Ask the model for a summary'''
        return self.model.get_summary(text, summarizer)

    def get_keywords(
        self,
        text,
        summarizer="Topics mentioned in the preceding paragraph include the following tags:"
        ):
        ''' Ask the model for keywords'''
        return self.model.get_keywords(text, summarizer)
