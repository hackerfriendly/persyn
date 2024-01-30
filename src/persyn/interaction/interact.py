'''
interact.py

The limbic system library.
'''

from dataclasses import dataclass
import random
from typing import Optional, Union, List, Tuple

import ulid
import requests

from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

from persyn.interaction import chrono

# Long and short term memory
from persyn.interaction.memory import Convo, Recall

# Prompt completion
from persyn.interaction.completion import LanguageModel

# Goals
from persyn.interaction.goals import Goal

# Color logging
from persyn.utils.color_logging import log
from persyn.utils.config import PersynConfig

rs = requests.Session()

@dataclass
class Interact:
    '''
    The Interact class contains all language handling routines and maintains
    the short and long-term memories for each service+channel.
    '''
    config: PersynConfig

    def __post_init__(self):
        self.lm = LanguageModel(self.config) # pylint: disable=invalid-name
        self.recall = Recall(self.config)
        self.goal = Goal(self.config)

    def send_chat(self, service: str, channel: str, msg: str, extra: Optional[str] = None) -> None:
        '''
        Send a chat message via the autobus.
        '''
        req = {
            "service": service,
            "channel": channel,
            "msg": msg,
            "extra": extra
        }

        try:
            reply = rs.post(f"{self.config.interact.url}/send_msg/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /send_msg/ to interact: {err}")
            return

    def template(self, context: Optional[str] = "") -> str:
        '''
        Return the current prompt template.

        Note that {kg} is used as a placeholder for knowledge graph memory but is never rendered.
        It's overridden later in status() to provide the full prompt context.

        Curly braces in the context are replaced with () to avoid template issues.
        '''
        return f"""It is {chrono.exact_time()} {chrono.natural_time()} on {chrono.today()}.
{self.config.interact.character}
""" + context.replace('{', '(').replace('}', ')') + """
{kg}
{history}

{human}: {input}
{bot}:"""

    def add_context(self, convo: Convo, raw=False) -> Union[str, List[Tuple[str, str]]]:
        ''' Add context to the prompt. If raw is True, return a list of tuples of (source, text). Otherwise return a string. '''
        log.warning(f"🧠 add_context: {convo.service}|{convo.channel} ({convo.id})")

        dialog = self.recall.convo_dialog(convo)

        context = []
        if not dialog:
            log.warning("🧠 ...without dialog.")
        else:
            log.warning("🧠 ...including goals and memories.")
            context += self.get_relevant_goals(convo, [('dialog', dialog)] + context)
            context += self.get_relevant_memories(convo, [('dialog', dialog)] + context)

        # Always retrieve recent summaries
        context += self.get_recent_summaries(convo, context)

        # TODO: Also fetch dialog from recently expired convos.

        # End with feels
        context += [self.get_sentiment_analysis(convo)]

        return context if raw else '\n'.join([ctx[1] for ctx in context])

    def get_sentiment_analysis(self, convo: Convo) -> Tuple[str, str]:
        ''' Fetch sentiment analysis for this convo '''
        sentiment = self.recall.fetch_convo_meta(convo.id, 'feels') or 'neutral'
        return (
            'sentiment analysis',
            f"{self.config.id.name}'s emotional state: {sentiment}."
        )

    def too_many_tokens(self, convo: Convo, text: str, used: Optional[int] = 0) -> bool:
        '''
        Count tokens like this: tokens in convo + tokens in text + an arbitrary number of tokens already used
        Return True if the token count is > than the fraction allowed by the config for chat_llm.

        TODO: Allow llm selection (currently always uses chat_llm)
        '''
        max_tokens = int(self.lm.max_prompt_length() * self.config.memory.context)
        history = self.recall.convo_dialog(convo)

        return self.lm.chat_llm.get_num_tokens(f"{history} {text}".strip()) + used > max_tokens # type: ignore

    def get_relevant_memories(self, convo: Convo, context: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str]]: # dangerous-default-value
        ''' Return a list of tuples of (source, text) for relevant memories '''
        if context is None:
            context = []

        relevant_memories = []
        related_convos = self.recall.find_related_convos(
            convo.service,
            convo.channel,
            '\n'.join([ctx[1] for ctx in context]),
            exclude_convo_ids=list(convo.visited),
            threshold=self.config.memory.relevance,
            size=5
        )
        used = len('\n'.join([ctx[1] for ctx in context]))
        for convo_id, score in related_convos:
            if convo_id in convo.visited:
                continue
            convo.visited.add(convo_id)
            summary = self.recall.fetch_summary(convo_id, final=True)
            if summary:
                if self.too_many_tokens(convo, summary + '\n'.join([ctx[1] for ctx in relevant_memories]), used):
                    break
                preamble = self.get_time_preamble(convo_id)
                relevant_memories.append((f"relevant memory ({score})", f"{self.config.id.name} recalls{preamble}\n{summary}"))
        return relevant_memories

    def get_relevant_goals(self, convo: Convo, context: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
        ''' Return a list of tuples of (source, text) for relevant goals '''
        if context is None:
            context = []

        relevant_goals = []
        used = len('\n'.join([ctx[1] for ctx in context]))
        for goal in self.goal.find_related_goals(
            convo.service,
            convo.channel,
            '\n'.join([ctx[1] for ctx in context]),
            threshold=self.config.memory.relevance,
            size=2
        ):
            actions = self.goal.list_actions(goal.goal_id)
            if not actions:
                continue
            # Actions are used only once. Delete them from the goal, but save for later evaluation.
            action = random.sample(actions, 1)[0]
            self.goal.undertake_action(convo.id, goal.goal_id, action)

            todo = f"{self.config.id.name}'s objective: {action}"
            if self.too_many_tokens(convo, todo, used):
                break
            log.warning(f"🎯 To achieve {goal.content}, {todo}")
            relevant_goals.append(("relevant goal", todo))
        return relevant_goals

    def get_recent_summaries(self, convo: Convo, context: Optional[List[Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
        ''' Return a list of tuples of (source, text) for recent summaries'''
        if context is None:
            context = []

        recent_summaries = []
        used = len('\n'.join([ctx[1] for ctx in context]))
        convo_ids = list(self.recall.list_convo_ids(convo.service, convo.channel, expired=True))

        for convo_id in convo_ids:
            summary = self.recall.fetch_summary(convo_id, final=True)
            if not summary:
                continue
            if self.too_many_tokens(convo, summary + '\n'.join([ctx[1] for ctx in recent_summaries]), used):
                break
            recent_summaries.append(("recent summary", f"{self.config.id.name} recalls{self.get_time_preamble(convo_id)} {summary}"))
        return recent_summaries

    def get_time_preamble(self, convo_id: str) -> str:
        ''' Return an appropriate time elapsed preamble '''
        last_ts = self.recall.id_to_timestamp(convo_id)
        if chrono.elapsed(last_ts) > 7200:
            return f" a conversation from {chrono.hence(last_ts)} ago:\n"
        return ""

    def save_summary(self, convo: Convo) -> None:
        '''
        Save the summary for this convo.
        While a convo is active, it contains the entire summary buffer (including most dialog).
        After the convo has expired, it contains only the final summary.
        '''
        convo.memories['redis'].add_texts(
            texts=[
                self.recall.convo_dialog(convo)
            ],
            metadatas=[
                {
                    "service": convo.service,
                    "channel": convo.channel,
                    "convo_id": convo.id,
                    "speaker_name": "narrator",
                    "verb": "summary",
                    "role": "bot"
                }
            ],
            keys=[
                f"{convo.id}:summary"
            ]
        )

    def retort(
        self,
        service: str,
        channel: str,
        msg: str,
        speaker_name: str,
        send_chat: bool = True,
        extra: Optional[str] = None
        ) -> str:
        '''
        Get a completion for the given channel.

        Returns the response. If send_chat is True, also send it to chat.
        '''
        log.info(f"💬 retort to: {msg}")

        convo = self.recall.fetch_convo(service, channel)
        if convo is None:
            convo = self.recall.new_convo(service, channel, speaker_name)

        prompt = PromptTemplate(
            input_variables=["input"],
            template=self.template(self.add_context(convo)),
            partial_variables={
                "human": speaker_name,
                "bot": self.config.id.name
            }
        )

        chain = ConversationChain(
            llm=self.lm.chat_llm,
            verbose=True,
            prompt=prompt,
            memory=convo.memories['combined']
        )

        # Hand it to langchain.
        reply = chain.invoke(input={'input':msg})['response']

        # trim() should probably be an output parser, but I can't make that work with memories.
        # So just rewrite history instead.
        trimmed = self.lm.trim(reply)
        if trimmed != reply:
            for mem in chain.memory.memories:
                if mem.chat_memory.messages:
                    mem.chat_memory.messages[-1].content = trimmed

        if send_chat:
            self.send_chat(service, channel, trimmed, extra)

        convo.memories['redis'].add_texts(
            texts=[
                msg,
                trimmed
            ],
            metadatas=[
                {
                    "service": service,
                    "channel": channel,
                    "convo_id": convo.id,
                    "speaker_name": speaker_name,
                    "verb": "dialog",
                    "role": "human"
                },
                {
                    "service": service,
                    "channel": channel,
                    "convo_id": convo.id,
                    "speaker_name": self.config.id.name,
                    "verb": "dialog",
                    "role": "bot"
                },
            ],
            keys=[
                f"{convo.id}:dialog:{str(ulid.ULID())}",
                f"{convo.id}:dialog:{str(ulid.ULID())}",
            ]
        )
        # Also save the summary
        self.save_summary(convo)

        log.info(f"💬 retort done: {reply}")
        return trimmed

    def status(self, service: str, channel: str, speaker_name: str) -> str:
        ''' Return the prompt and chat history for this channel '''
        convo = self.recall.fetch_convo(service, channel)
        if convo is None:
            convo = self.recall.new_convo(service, channel, speaker_name)

        prompt = PromptTemplate(
            input_variables=["input"],
            template=self.template(self.add_context(convo)),
            partial_variables={
                "human": convo.memories['summary'].human_prefix,
                "bot": self.config.id.name
            }
        )

        return prompt.format(
            kg='', # FIXME: this is a hack to avoid rendering {kg} in the template
            history=self.recall.convo_dialog(convo),
            input='input'
        )

    def inject_idea(self, service: str, channel: str, idea: str, verb: str="recalls") -> bool:
        '''
        Directly inject an idea into recall memory.
        '''
        log.info(f"💉 inject_idea: {service}|{channel} ({verb}) {idea[:20]}…")

        if verb == 'dialog':
            log.error("💉 inject_idea: dialog is handled by retort(), skipping.")
            return False

        convo = self.recall.fetch_convo(service, channel)
        if convo is None:
            return False

        log.debug(self.recall.convo_dialog(convo))

        convo.memories['combined'].save_context({"input": ""}, {"output": f"({verb}) {idea}"})
        self.save_summary(convo)

        return True

    def summarize_channel(self, service: str, channel: str, convo_id: Optional[str] = None, final: Optional[bool] = False) -> str:
        ''' Summarize a channel in a few sentences. '''
        if convo_id is None:
            convo_id = self.recall.get_last_convo_id(service, channel)

        if convo_id is None:
            return ""

        summary = self.lm.summarize_text(self.recall.fetch_summary(convo_id), final=final)

        if final and summary:
            log.info(f"🎬 Saving final summary for {service}|{channel} : {convo_id}")
            self.recall.redis.hset(f"{self.recall.convo_prefix}:{convo_id}:summary", "final", summary)

        return summary
