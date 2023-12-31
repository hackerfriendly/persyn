'''
interact.py

The limbic system library.
'''
from dataclasses import dataclass
import random

import ulid
import requests

from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.redis import RedisTag

from persyn.interaction import chrono

# Long and short term memory
from persyn.interaction.memory import Recall

# Prompt completion
from persyn.interaction.completion import LanguageModel

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

        # Pick a language model for completion
        self.lm = LanguageModel(self.config) # pylint: disable=invalid-name

        # Then create the Recall object (conversation management)
        self.recall = Recall(self.config)

        # More Langchain

        # agent_tools = []
        # zim / kiwix data sources
        # if self.config.get('zimx'):
        #     for cfgtool in self.config.zim:
        #         log.info("💿 Loading zim:", cfgtool)
        #         zim = Tool(
        #                 name=str(cfgtool),
        #                 func=ZimWrapper(path=self.config.zim.get(cfgtool).path).run,
        #                 description=self.config.zim.get(cfgtool).description
        #             )
        #         agent_tools.append(zim)

        # if self.config.lm.get('anthropic_key'):
        #     def ask_claude(query: str) -> str:
        #         """Ask Claude for help"""
        #         return self.lm.ask_claude(str)
        #     claude = Tool.from_function(
        #         name="Claude",
        #         func=ask_claude,
        #         description="Ask Claude a question. Claude can check facts, do math, and offer general advice."
        #     )
        #     agent_tools.append(claude)

        # Other tools: introspection (assess software + hardware), Claude (ask for facts)

        # self.agent = initialize_agent(
        #     agent_tools, self.llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
        # )

    def send_chat(self, service, channel, msg):
        '''
        Send a chat message via the autobus.
        '''
        req = {
            "service": service,
            "channel": channel,
            "msg": msg
        }

        try:
            reply = rs.post(f"{self.config.interact.url}/send_msg/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /send_msg/ to interact: {err}")
            return

    def template(self, context="") -> str:
        '''
        Return the current prompt template.

        Note that {kg} is used as a placeholder for knowledge graph memory but is never rendered.
        It's overridden later in status() to provide the full prompt context.
        '''
        return f"""It is {chrono.exact_time()} {chrono.natural_time()} on {chrono.today()}.
{self.config.interact.character}
""" + context + """
{kg}
{history}

{human}: {input}
{bot}:"""

    def add_context(self, convo):
        ''' Additional context for the prompt '''
        context = []

        # Sentiment analysis
        context.append(f"{self.config.id.name} is feeling {self.recall.fetch_convo_meta(convo.id, 'feels') or 'nothing in particular'}.")

        # Relevant memories
        # Available metadata: service, channel, role, speaker_name, verb
        summaries_only = RedisTag("verb") == "summary"
        this_channel = (RedisTag("service") == convo.service) & (RedisTag("channel") == convo.channel)

        rds = convo.memories['redis']

        ret = rds.similarity_search_with_score(
            self.current_dialog(convo),
            k=20,
            distance_threshold=self.config.memory.relevance,
            # FIXME: applying a filter and distance_threshold throws ResponseError: Invalid attribute yield_distance_as
            # filter=this_channel
        )
        log.warning(ret)

        # TODO: Instead of random, weight by age and relevance a la Stanford Smallville
        while ret:
            doc = random.sample(ret, k=1)[0]
            ret.remove(doc)

            convo_id = doc[0].metadata['id'].split(':')[3]
            if convo_id in convo.visited:
                log.warning(f"🤌 Already visited this convo:", convo_id)
                continue

            convo.visited.add(convo_id)
            summary = self.recall.fetch_summary(convo_id)
            log.warning("✅ Found relevant memory with score:", doc[1])
            context.append(summary)
            break

        # Recent summaries

        # Recent conversation

        return '\n'.join(context)

    def current_dialog(self, convo) -> str:
        ''' Return the current dialog from convo '''
        return convo.memories['summary'].load_memory_variables({})['history'].lstrip("System: ")

    def retort(self, service, channel, msg, speaker_name, send_chat=True):  # pylint: disable=too-many-locals
        '''
        Get a completion for the given channel.

        Returns the response. If send_chat is True, also send it to chat.
        '''
        msg_id = str(ulid.ULID())
        log.info(f"💬 get_reply to: {msg}")

        convo = self.recall.get_convo(service, channel)
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
        reply = chain.run(input=msg)
        reply_id = str(ulid.ULID())

        # trim() should probably be an output parser, but I can't make that work with memories.
        # So just rewrite history instead.
        trimmed = self.lm.trim(reply)
        if trimmed != reply:
            for mem in chain.memory.memories:
                if mem.chat_memory.messages:
                    mem.chat_memory.messages[-1].content = trimmed

        if send_chat:
            self.send_chat(service, channel, trimmed)

        convo.memories['redis'].add_texts(
            texts=[
                msg,
                trimmed,
                self.current_dialog(convo)
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
                {
                    "service": service,
                    "channel": channel,
                    "convo_id": convo.id,
                    "speaker_name": "narrator",
                    "verb": "summary",
                    "role": "bot"
                }
            ],
            keys=[
                f"{convo.id}:lines:{msg_id}",
                f"{convo.id}:lines:{reply_id}",
                f"{convo.id}:summary"
            ]
        )

        # vvvv OLD CODE BELOW vvvv

        # # convo_id = self.recall.convo_id(service, channel)
        # convo = self.recall.convo(service, channel, feels=True)

        # prompt = self.generate_prompt(service, channel)

        # # TODO: vvv  Use this time to backfill context!  vvv

        # # Ruminate a bit
        # entities = self.extract_entities(msg)

        # if entities:
        #     log.warning(f"🆔 extracted entities: {entities}")
        # else:
        #     entities = self.extract_nouns('\n'.join(convo))[:8]
        #     log.warning(f"🆔 extracted nouns: {entities}")

        # # Reflect on this conversation
        # visited = self.gather_memories(service, channel, entities)
        # summaries = []
        # for doc in self.recall.summaries(service, channel, None, size=1, raw=True):
        #     if doc.convo_id not in visited and doc.summary not in summaries:
        #         log.warning("💬 Adding summary:", doc.summary)
        #         summaries.append(doc.summary)
        #         visited.append(doc.convo_id)

        # ^^^  end TODO  ^^^

        log.info(f"💬 get_reply done: {reply}")

        return trimmed


    def status(self, service, channel, speaker_name) -> str:
        ''' Return the prompt and chat history for this channel '''
        convo = self.recall.get_convo(service, channel)
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
            kg='', #convo.memories['summary'].moving_summary_buffer,
            history=convo.memories['summary'].load_memory_variables({})['history'],
            input='input'
        )

    def extract_nouns(self, text):
        ''' return a list of all nouns (except pronouns) in text '''
        doc = self.lm.nlp(text)
        nouns = {
            n.text.strip()
            for n in doc.noun_chunks
            if n.text.strip() != self.config.id.name
            for t in n
            if t.pos_ != 'PRON'
        }
        return list(nouns)

    def extract_entities(self, text):
        ''' return a list of all entities in text '''
        doc = self.lm.nlp(text)
        return list({n.text.strip() for n in doc.ents if n.text.strip() != self.config.id.name})

    def inject_idea(self, service, channel, idea, verb="recalls"):
        '''
        Directly inject an idea into recall memory.
        '''
        convo = self.recall.get_convo(service, channel)
        if convo is None:
            # So we're talking to ourself?
            convo = self.recall.new_convo(service, channel, self.config.id.name)

        # Add to summary memory. Dialog is automatically handled by langchain.
        if verb != 'dialog':
            convo.memories['summary'].chat_memory.add_ai_message(f"({verb}) {idea}")

            # Add to redis
            convo.memories['redis'].add_texts(
                texts=[idea],
                metadatas=[
                    {
                        "service": service,
                        "channel": channel,
                        "convo_id": convo.id,
                        "speaker_name": self.config.id.name,
                        "verb": verb
                    }
                ],
                keys=[f"{convo.id}:{str(ulid.ULID())}"]
            )

    def summarize_channel(self, service, channel, convo_id=None) -> str:
        ''' Summarize a channel in a few sentences. '''
        if convo_id is None:
            convo_id = self.recall.get_last_convo_id(service, channel)

        if convo_id is None:
            return ""

        return self.lm.summarize_text(self.recall.fetch_summary(convo_id))

        # if verb != "decides" and idea in '\n'.join(self.recall.convo(service, channel, feels=True)):
        #     log.warning("🤌  Already had this idea, skipping:", idea)
        #     return

        # self.recall.save_convo_line(
        #     service,
        #     channel,
        #     msg=idea,
        #     speaker_name=self.config.id.name,
        #     convo_id=self.recall.convo_id(service, channel),
        #     verb=verb
        # )

        # log.warning(f"🤔 {verb}:", idea)
        # return

    # def surmise(self, service, channel, topic, size=10):
    #     ''' Stub for recall '''
    #     return self.recall.surmise(service, channel, topic, size)

    # def add_goal(self, service, channel, goal):
    #     ''' Stub for recall '''
    #     return self.recall.add_goal(service, channel, goal)

    # def get_goals(self, service, channel, goal=None, size=10):
    #     ''' Stub for recall '''
    #     return self.recall.get_goals(service, channel, goal, size)

    # def list_goals(self, service, channel, size=10):
    #     ''' Stub for recall '''
    #     return self.recall.list_goals(service, channel, size)

    # def read_news(self, service, channel, url, title):
    #     ''' Let's check the news while we ride the autobus. '''
    #     req = {
    #         "service": service,
    #         "channel": channel,
    #         "url": url,
    #         "title": title
    #     }
    #     try:
    #         reply = rs.post(f"{self.config.interact.url}/read_news/", params=req, timeout=10)
    #         reply.raise_for_status()
    #     except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
    #         log.critical(f"🤖 Could not post /read_news/ to interact: {err}")

    # def read_url(self, service, channel, url):
    #     ''' Let's ride the autobus on the information superhighway. '''
    #     parsed = urlparse(url)
    #     if not bool(parsed.scheme) and bool(parsed.netloc):
    #         log.warning("👨‍💻 Not a URL:", url)
    #         return

    #     req = {
    #         "service": service,
    #         "channel": channel,
    #         "url": url
    #     }
    #     try:
    #         reply = rs.post(f"{self.config.interact.url}/read_url/", params=req, timeout=10)
    #         reply.raise_for_status()
    #     except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
    #         log.critical(f"🤖 Could not post /read_url/ to interact: {err}")


    # def get_generic_agent_reply(self, prompt: str):
    #     ''' Try to take an action using the agent '''
    #     try:
    #         ret = self.agent.run(prompt)
    #     except Exception as err:
    #         log.error("🕵️‍♂️  Agent generated an exception, skipping:", err)
    #         return None

    #     if ret == "Agent stopped due to iteration limit or time limit.":
    #         log.warning("🕵️‍♂️ ", ret)
    #         return None

    #     log.warning("🕵️‍♂️  Found an answer:", ret)
    #     return ret

    # def validate_agent_reply(self, agent_reply):
    #     ''' Returns True if the question was answered, otherwise False. '''
    #     if (
    #         not agent_reply
    #         or agent_reply == "Agent stopped due to iteration limit or time limit"
    #     ):
    #         return False

    #     # TODO: encode "Yes" and "No", then ask llm if the question was answered

    #     negatives = ["I cannot", "I can't", "I could not", "I couldn't", "is not mentioned", "isn't mentioned", "unable"]
    #     if any(term in agent_reply for term in negatives):
    #         return False

    #     return True

    # def try_the_agent(self, prompt, service, channel):
    #     '''
    #     Let the langchain agent weigh in by injecting thoughts.
    #     Agents do not speak directly.
    #     '''

    #     log.warning("🕵️  Consult the agent.")
    #     reply = self.get_generic_agent_reply(prompt)

    #     if self.validate_agent_reply(reply):
    #         log.warning("🕵️  Generic agent success:", reply)
    #         self.inject_idea(
    #             service,
    #             channel,
    #             reply,
    #             verb="thinks"
    #         )
    #     else:
    #         log.warning("🤷 Generic agent was no help.")


    def gather_memories(self, service, channel, entities, visited=None):
        '''
        Look for relevant convos and summaries using memory, relationship graphs, and entity matching.
        '''
        # TODO: weigh retrievals with "importance" and recency, a la Stanford Smallville
        return []

        # if visited is None:
        #     visited = []

        # convo = self.recall.convo(service, channel, feels=False)

        # if not convo:
        #     return visited

        # ranked = self.recall.find_related_convos(
        #     service, channel,
        #     query='\n'.join(convo[:5]),
        #     size=10,
        #     current_convo_id=self.recall.convo_id(service, channel),
        #     threshold=self.config.memory.relevance
        # )

        # # No hits? Don't try so hard.
        # if not ranked:
        #     log.warning("🍸 Nothing relevant. Try lateral thinking.")
        #     ranked = self.recall.find_related_convos(
        #         service, channel,
        #         query='\n'.join(convo),
        #         size=1,
        #         current_convo_id=self.recall.convo_id(service, channel),
        #         threshold=self.config.memory.relevance * 1.4,
        #         any_convo=True
        #     )

        # for hit in ranked:
        #     if hit.convo_id not in visited:
        #         if hit.service == 'import_service':
        #             log.info("📚 Hit found from import:", hit.channel)
        #         the_summary = self.recall.get_summary_by_id(hit.convo_id)
        #         # Hit a sentence? Inject the summary.
        #         # if the_summary:
        #         #     self.inject_idea(
        #         #         service, channel,
        #         #         f"{the_summary.summary} In that conversation, {hit.speaker_name} said: {hit.msg}",
        #         #         verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
        #         #     )
        #         if the_summary:
        #             self.inject_idea(
        #                 service, channel,
        #                 the_summary.summary,
        #                 verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
        #             )
        #         # No summary? Just inject the sentence.
        #         else:
        #             self.inject_idea(
        #                 service, channel,
        #                 f"{hit.speaker_name} said: {hit.msg}",
        #                 verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
        #             )
        #         visited.append(hit.convo_id)
        #         log.info(f"🧵 Related convo {hit.convo_id} ({float(hit.score):0.3f}):", hit.msg)

        # # Look for other summaries that match detected entities
        # if entities:
        #     visited = self.gather_summaries(service, channel, entities, size=2, visited=visited)

        # return visited

    # def gather_summaries(self, service, channel, entities, size, visited=None):
    #     '''
    #     If a previous convo summary matches entities and seems relevant, inject its memory.

    #     Returns a list of ids of summaries injected.
    #     '''
    #     if not entities:
    #         return []

    #     if visited is None:
    #         visited = []

    #     search_term = ' '.join(entities)
    #     log.warning(f"ℹ️  look up '{search_term}' in memories")

    #     for summary in self.recall.summaries(service, channel, search_term, size=10, raw=True):
    #         if summary.convo_id in visited:
    #             continue
    #         visited.append(summary.convo_id)

    #         log.warning(f"🐘 Memory found: {summary.summary}")
    #         self.inject_idea(service, channel, summary.summary, "remembers")

    #         if len(visited) >= size:
    #             break

    #     return visited

    # def gather_opinions(self, service, channel, entities):
    #     '''
    #     Gather opinions from memory.

    #     This happens asynchronously via the event bus, so opinions might not be immediately available
    #     for conversation.
    #     '''
    #     if not entities:
    #         return

    #     the_sample = random.sample(entities, k=min(3, len(entities)))

    #     req = {
    #         "service": service,
    #         "channel": channel,
    #         "entities": the_sample
    #     }

    #     log.warning(f"🧷 opine : {the_sample}")
    #     try:
    #         reply = rs.post(f"{self.config.interact.url}/opine/", params=req, timeout=10)
    #         reply.raise_for_status()
    #     except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
    #         log.critical(f"🤖 Could not post /opine/ to interact: {err}")
    #         return

    # def check_goals(self, service, channel, convo):
    #     ''' Have we achieved our goals? '''
    #     goals = self.recall.list_goals(service, channel)

    #     if goals:
    #         req = {
    #             "service": service,
    #             "channel": channel,
    #             "convo": '\n'.join(convo),
    #             "goals": goals
    #         }

    #         try:
    #             reply = rs.post(f"{self.config.interact.url}/check_goals/", params=req, timeout=10)
    #             reply.raise_for_status()
    #         except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
    #             log.critical(f"🤖 Could not post /check_goals/ to interact: {err}")

    # def get_feels(self, service, channel, convo_id, room):
    #     ''' How are we feeling? Let's ask the autobus. '''
    #     req = {
    #         "service": service,
    #         "channel": channel,
    #         "convo_id": convo_id,
    #     }
    #     data = {
    #         "room": room
    #     }

    #     try:
    #         reply = rs.post(f"{self.config.interact.url}/vibe_check/", params=req, data=data, timeout=10)
    #         reply.raise_for_status()
    #     except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
    #         log.critical(f"🤖 Could not post /vibe_check/ to interact: {err}")

    # def save_knowledge_graph(self, service, channel, convo_id, convo):
    #     ''' Build a pretty graph of this convo... via the autobus! '''
    #     req = {
    #         "service": service,
    #         "channel": channel,
    #         "convo_id": convo_id,
    #     }
    #     data = {
    #         "convo": convo
    #     }

    #     try:
    #         reply = rs.post(f"{self.config.interact.url}/build_graph/", params=req, data=data, timeout=10)
    #         reply.raise_for_status()
    #     except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
    #         log.critical(f"🤖 Could not post /build_graph/ to interact: {err}")
