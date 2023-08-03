'''
interact.py

The limbic system library.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order
import random
import re # used by custom filters

from urllib.parse import urlparse
from typing import Optional, List

import requests

from langchain import LLMMathChain, OpenAI
from langchain.agents.agent_toolkits import create_vectorstore_router_agent, VectorStoreInfo
from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool, Tool, tool
from langchain.tools.vectorstore.tool import VectorStoreQATool, VectorStoreQAWithSourcesTool
from langchain.vectorstores import Redis

from pydantic import Field

# Long and short term memory
from persyn.interaction.memory import Recall

# Time handling
from persyn.interaction.chrono import exact_time, natural_time, ago, today, elapsed, get_cur_ts

# Prompt completion
from persyn.interaction.completion import LanguageModel

# Custom langchain
from persyn.langchain.zim import ZimWrapper

# Color logging
from persyn.utils.color_logging import log

def doiify(text):
    ''' Turn DOIs into doi.org links '''
    ret = re.sub(r'\b(10.\d{4,9}/[-._;()/:a-zA-Z0-9]+)\b', r'http://doi.org/\1', text).rstrip('.')
    return re.sub(r'http://doi.org/http://doi.org/', r'http://doi.org/', ret)

class MyVectorStoreRouterToolkit(BaseToolkit):
    """Toolkit for routing between vector stores."""

    vectorstores: List[VectorStoreInfo] = Field(exclude=True)
    llm: BaseLanguageModel = Field(default_factory=lambda: OpenAI(temperature=0))
    tools: Optional[List[BaseTool]] = []

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools: List[BaseTool] = []
        for vectorstore_info in self.vectorstores:
            description = VectorStoreQATool.get_description(
                vectorstore_info.name, vectorstore_info.description
            )
            qa_tool = VectorStoreQAWithSourcesTool(
                name=vectorstore_info.name,
                description=description,
                vectorstore=vectorstore_info.vectorstore,
                llm=self.llm,
            )
            tools.append(qa_tool)
        return tools + self.tools

@tool("None", return_direct=False)
def we_are_done(query: str) -> str:
    """Time to finish up."""
    return "I now have all the information requested in the original question."

@tool(return_direct=True)
def take_a_photo(query: str) -> str:
    """ Think of anything and instantly take a photo of whatever you imagine. """
    return f"photo:{query}"

class Interact():
    '''
    The Interact class contains all language handling routines and maintains
    the short and long-term memories for each service+channel.
    '''
    def __init__(self, persyn_config):
        self.config = persyn_config

        # Pick a language model for completion
        self.completion = LanguageModel(config=persyn_config)

        # Identify a custom reply filter, if any. Should be a Python expression
        # that receives the reply as the variable 'reply'. (Also has access to
        # 'self' technically and anything else loaded at module scope, so be
        # careful.)
        self.custom_filter = None
        if hasattr(self.config.interact, "filter"):
            assert re # prevent "unused import" type linting
            self.custom_filter = eval(f"lambda reply: {self.config.interact.filter}") # pylint: disable=eval-used

        # Then create the Recall object (short-term, long-term, and graph memory).
        self.recall = Recall(persyn_config)

        # Langchain
        self.llm = ChatOpenAI(
            temperature=self.config.completion.temperature,
            model=self.config.completion.completion_model,
            openai_api_key=self.config.completion.api_key
        )
        embeddings = OpenAIEmbeddings(openai_api_key=self.config.completion.api_key)

        # langchain tools
        agent_tools = [
            Tool(
                name="Calculator",
                func=LLMMathChain.from_llm(llm=self.llm, verbose=True).run,
                description="Answer simple math questions accurately. Inputs should be simple expressions, like 2+2. This tool does not accept python code.",
                return_direct=True
            ),
            Tool(
                    name="Say something",
                    func=lambda x: f"say:{x}",
                    description="Continue the conversation",
                    return_direct=True
            ),
            we_are_done,
            take_a_photo
        ]

        if self.config.get('zim'):
            for cfgtool in self.config.zim:
                log.info("💿 Loading zim:", cfgtool)
                agent_tools.append(
                    Tool(
                        name=str(cfgtool),
                        func=ZimWrapper(path=self.config.zim.get(cfgtool).path).run,
                        description=self.config.zim.get(cfgtool).description
                    )
                )

        vectorstores = []
        if self.config.get('vectorstores'):
            for vs in self.config.vectorstores:
                log.info("💾 Loading vectorstore:", vs)

                vectorstore = Redis(
                    redis_url=persyn_config.memory.redis,
                    index_name=self.config.vectorstores.get(vs).index,
                    embedding_function=embeddings.embed_query,
                )

                vectorstores.append(VectorStoreInfo(
                    name=self.config.vectorstores.get(vs).name,
                    description=self.config.vectorstores.get(vs).description,
                    vectorstore=vectorstore
                ))

        router_toolkit = MyVectorStoreRouterToolkit(
            vectorstores=vectorstores,
            llm=self.llm,
            tools=agent_tools
        )

        self.agent_executor = create_vectorstore_router_agent(
            llm=self.llm, toolkit=router_toolkit, verbose=True,
            prefix="""You are an agent designed to answer questions.
            You have access to tools for interacting with different sources, and the inputs to the tools are questions.
            For complex questions, you can break the question down into sub questions and use tools to answers the sub questions.
            Use whichever tool is relevant for answering the question at hand. If a tool is not relevant, do not use it.
            If no tools are relevant, finish with "Thought:I now know the final answer."
            If you don't have a final answer, finish with "Final Answer:" and a suitable reason why you could not answer.
            """,
            agent_executor_kwargs = {
                'max_iterations': 10,
                'max_execution_time': 60,
                'handle_parsing_errors': True
            }
        )

        self.enc = self.completion.model.get_enc()

    def summarize_convo(
        self,
        service,
        channel,
        save=True,
        include_keywords=False,
        context_lines=0,
        dialog_only=True,
        model=None,
        convo_id=None,
        save_kg=True
        ):
        '''
        Generate a summary of the current conversation for this channel.
        Also generate and save opinions about detected topics.

        If save == True, save convo to long term memory and generate
        knowledge graph nodes (via the autobus).

        Returns the text summary.
        '''
        if convo_id is None:
            convo_id = self.recall.convo_id(service, channel)
        if not convo_id:
            return ""

        if dialog_only:
            text = self.recall.convo(service, channel, verb='dialog') or self.recall.summaries(service, channel, size=3)
        else:
            text = self.recall.convo(service, channel, feels=True)

        if not text:
            return ""

        log.warning("∑ summarizing convo")

        convo_text = '\n'.join(text)

        log.info(convo_text)

        summary = self.completion.get_summary(
            text=convo_text,
            summarizer=f"Briefly summarize this conversation from {self.config.id.name}'s point of view, and convert pronouns and verbs to the first person."
        )
        keywords = self.completion.get_keywords(summary)

        if save:
            self.recall.save_summary(service, channel, convo_id, summary, keywords)

        if save_kg:
            self.save_knowledge_graph(service, channel, convo_id, convo_text)

        if include_keywords:
            return summary + f"\nKeywords: {keywords}"

        if context_lines:
            return "\n".join(text[-context_lines:] + [summary])

        return summary

    def get_vector_agent_reply(self, context):
        ''' Run the agent executor'''
        convo = '\n'.join(context[:-1])
        query = context[-1]
        return doiify(self.agent_executor.run(f"""
            Examine the following conversation:
            {convo}

            Using only the tools available to you, answer this question:
            {query}

            Cite your sources including the doi if available.
        """
    )).strip()

    def validate_agent_reply(self, agent_reply):
        ''' Returns True if the question was answered, otherwise False. '''
        if (
            not agent_reply
            or len(agent_reply) < 40
            or agent_reply == "Agent stopped due to iteration limit or time limit"
        ):
            return False

        # TODO: encode "Yes" and "No", then ask llm if the question was answered

        negatives = ["tool", "I cannot", "I can't", "I could not", "I couldn't", "is not mentioned", "isn't mentioned", "unable"]
        if any(term in agent_reply for term in negatives):
            return False

        return True

    def gather_memories(self, service, channel, entities, visited=None):
        '''
        Look for relevant convos and summaries using memory, relationship graphs, and entity matching.

        TODO: weigh retrievals with "importance" and recency, a la Stanford Smallville
        '''
        if visited is None:
            visited = []

        convo = self.recall.convo(service, channel, feels=True)

        if not convo:
            return visited

        ranked = self.recall.find_related_convos(
            service, channel,
            query='\n'.join(convo[:5]),
            size=10,
            current_convo_id=self.recall.convo_id(service, channel),
            threshold=self.config.memory.relevance
        )

        # No hits? Don't try so hard.
        if not ranked:
            log.warning("🍸 Nothing relevant. Try lateral thinking.")
            ranked = self.recall.find_related_convos(
                service, channel,
                query='\n'.join(convo),
                size=1,
                current_convo_id=self.recall.convo_id(service, channel),
                threshold=self.config.memory.relevance * 1.4,
                any_convo=True
            )

        if not ranked:
            log.warning("💁‍♂️ Vicarious comprehension")
            self.inject_idea(
                service, channel,
                random.choice([
                    f"{self.config.id.name} can't recall a specific relevant experience.",
                    f"that the extent of {self.config.id.name}'s familiarity with the subject comes only from online sources, not direct participation.",
                    f"{self.config.id.name}'s understanding of the issue is academic, not experiential.",
                    f"they can theorize about the topic, but can't draw upon any personal experiences to validate {self.config.id.name}'s views.",
                    f"the anecdotes they've heard and the articles they've read are {self.config.id.name}'s only source of knowledge on the topic.",
                    f"that {self.config.id.name} has only circumstantial knowledge about the topic, not personal insights.",
                    f"that while they can offer informed opinions, {self.config.id.name} hasn't had a relevant direct experience.",
                ]),
                verb="realizes"
            )

        for hit in ranked:
            if hit.convo_id not in visited:
                if hit.service == 'import_service':
                    log.info("📚 Hit found from import:", hit.channel)
                the_summary = self.recall.get_summary_by_id(hit.convo_id)
                # Hit a sentence? Inject the summary and the sentence.
                if the_summary:
                    self.inject_idea(
                        service, channel,
                        f"{the_summary.summary} In that conversation, {hit.speaker_name} said: {hit.msg}",
                        verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
                    )
                # No summary? Just inject the sentence.
                else:
                    self.inject_idea(
                        service, channel,
                        f"{hit.speaker_name} said: {hit.msg}",
                        verb=f"remembers that {ago(self.recall.entity_id_to_timestamp(hit.convo_id))} ago"
                    )
                visited.append(hit.convo_id)
                log.info(f"🧵 Related convo {hit.convo_id} ({float(hit.score):0.3f}):", hit.msg)

        # Look for other summaries that match detected entities
        if entities:
            visited = self.gather_summaries(service, channel, entities, size=2, visited=visited)

        return visited

    def gather_summaries(self, service, channel, entities, size, visited=None):
        '''
        If a previous convo summary matches entities and seems relevant, inject its memory.

        Returns a list of ids of summaries injected.
        '''
        if not entities:
            return []

        if visited is None:
            visited = []

        search_term = ' '.join(entities)
        log.warning(f"ℹ️  look up '{search_term}' in memories")

        for summary in self.recall.summaries(service, channel, search_term, size=10, raw=True):
            if summary.convo_id in visited:
                continue
            visited.append(summary.convo_id)

            log.warning(f"🐘 Memory found: {summary.summary}")
            self.inject_idea(service, channel, summary.summary, "remembers")

            if len(visited) >= size:
                break

        return visited

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
            reply = requests.post(f"{self.config.interact.url}/send_msg/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /send_msg/ to interact: {err}")
            return

    def generate_photo(self, service, channel, prompt):
        ''' Generate a photo and send it to a channel '''
        req = {
            "channel": channel,
            "service": service,
            "prompt": prompt,
            "width": 1024,
            "height": 512,
            "bot_name": self.config.id.name,
            "bot_id": self.config.id.guid
        }
        try:
            reply = requests.post(f"{self.config.dreams.url}/generate/", params=req, timeout=10)
            if reply.ok:
                log.warning(f"{self.config.dreams.url}/generate/", f"{prompt}: {reply.status_code}")
            else:
                log.error(f"{self.config.dreams.url}/generate/", f"{prompt}: {reply.status_code} {reply.json()}")
            return reply.ok
        except requests.exceptions.ConnectionError as err:
            log.error(f"{self.config.dreams.url}/generate/", err)
            return False


    def gather_facts(self, service, channel, entities):
        '''
        Gather facts (from Wikipedia) and opinions (from memory).

        This happens asynchronously via the event bus, so facts and opinions
        might not be immediately available for conversation.
        '''
        if not entities:
            return

        the_sample = random.sample(entities, k=min(3, len(entities)))

        req = {
            "service": service,
            "channel": channel,
            "entities": the_sample
        }

        for endpoint in ['opine']:
            log.warning(f"🧾 {endpoint} : {the_sample}")
            try:
                reply = requests.post(f"{self.config.interact.url}/{endpoint}/", params=req, timeout=10)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /{endpoint}/ to interact: {err}")
                return

    def check_goals(self, service, channel, convo):
        ''' Have we achieved our goals? '''
        goals = self.recall.list_goals(service, channel)

        if goals:
            req = {
                "service": service,
                "channel": channel,
                "convo": '\n'.join(convo),
                "goals": goals
            }

            try:
                reply = requests.post(f"{self.config.interact.url}/check_goals/", params=req, timeout=10)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /check_goals/ to interact: {err}")

    def get_feels(self, service, channel, convo_id, room):
        ''' How are we feeling? Let's ask the autobus. '''
        req = {
            "service": service,
            "channel": channel,
            "convo_id": convo_id,
        }
        data = {
            "room": room
        }

        try:
            reply = requests.post(f"{self.config.interact.url}/vibe_check/", params=req, data=data, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /vibe_check/ to interact: {err}")

    def save_knowledge_graph(self, service, channel, convo_id, convo):
        ''' Build a pretty graph of this convo... via the autobus! '''
        req = {
            "service": service,
            "channel": channel,
            "convo_id": convo_id,
        }
        data = {
            "convo": convo
        }

        try:
            reply = requests.post(f"{self.config.interact.url}/build_graph/", params=req, data=data, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /build_graph/ to interact: {err}")

    # Need to instrument this. It takes far too long and isn't async.
    def get_reply(self, service, channel, msg, speaker_name, speaker_id, send_chat=True):  # pylint: disable=too-many-locals
        '''
        Get the best reply for the given channel. Saves to recall memory.

        Returns the best available reply. If send_chat is True, also send it to chat.
        '''
        log.info(f"💬 get_reply to: {msg}")

        convo_id = self.recall.convo_id(service, channel)

        if msg != '...':
            self.recall.save_convo_line(
                service,
                channel,
                msg,
                speaker_name,
                speaker_id,
                convo_id=convo_id,
                verb='dialog'
            )

        convo = self.recall.convo(service, channel, feels=True)
        last_sentence = None

        if convo:
            last_sentence = convo.pop()

        # TODO: Move this to CNS
        # Save the knowledge graph every 5 lines
        # if convo and len(convo) % 5 == 0:
        #     self.save_knowledge_graph(service, channel, convo_id, convo)

        # Ruminate a bit
        entities = self.extract_entities(msg)

        if entities:
            log.warning(f"🆔 extracted entities: {entities}")
        else:
            entities = self.extract_nouns('\n'.join(convo))[:8]
            log.warning(f"🆔 extracted nouns: {entities}")

        # Reflect on this conversation
        visited = self.gather_memories(service, channel, entities)

        # Facts and opinions
        self.gather_facts(service, channel, entities)

        # TODO: Move this to CNS
        # Goals. Don't give out _too_ many trophies.
        if random.random() < 0.5:
            self.check_goals(service, channel, convo)

        # Our mind might have been wandering, so remember the last thing that was said.
        if last_sentence:
            convo.append(last_sentence)

        summaries = []
        for doc in self.recall.summaries(service, channel, None, size=5, raw=True):
            if doc.convo_id not in visited and doc.summary not in summaries:
                log.warning("💬 Adding summary:", doc.summary)
                summaries.append(doc.summary)
                visited.append(doc.convo_id)

        lts = self.recall.get_last_timestamp(service, channel)
        convo = self.recall.convo(service, channel, feels=True)
        prompt = self.generate_prompt(summaries, convo, service, channel, lts)

        # Is this just too much to think about?
        if (self.completion.toklen(prompt) * 0.9) > self.completion.max_prompt_length():
            # Kick off a summary request via autobus. Yes, we're talking to ourselves now.
            log.warning("🥱 get_reply(): prompt too long, summarizing.")
            req = {
                "service": service,
                "channel": channel,
                "save": True
            }
            try:
                reply = requests.post(f"{self.config.interact.url}/summary/", params=req, timeout=60)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /summary/ to interact: {err}")
                return " :dancer: :interrobang: "

            convo = self.recall.convo(service, channel, feels=True)
            prompt = self.generate_prompt([], convo, service, channel, lts)

        log.warning("🕵️‍♀️  Consult the vector agent.")
        reply = self.get_vector_agent_reply(convo)
        if self.validate_agent_reply(reply):
            log.warning("🕵️‍♀️  Agent success:", reply)
            self.inject_idea(service, channel, idea=reply, verb="thinks")
            # self.send_chat(service, channel, reply)

            # # next reply will continue the thought
            # convo[-1] = f"{self.config.id.name}: {reply}"

        else:
            self.inject_idea(service, channel, reply, verb="thinks")
            log.warning("🤷 Agent was no help.")

            convo = self.recall.convo(service, channel, feels=True)
            prompt = self.generate_prompt(summaries, convo, service, channel, lts)
            reply = self.completion.get_reply(prompt)

        if self.custom_filter:
            try:
                reply = self.custom_filter(reply)
            except Exception as err: # pylint: disable=broad-except
                log.warning(f"🤮 Custom filter failed: {err}")

        # Say it!
        if send_chat:
            self.send_chat(service, channel, reply)

        ## TODO: move these to CNS
        # Sentiment analysis via the autobus
        self.get_feels(service, channel, convo_id, f'{prompt} {reply}')

        if 'http' in msg:
            # Regex chosen by GPT-4. 😵‍💫
            for url in re.findall(r'http[s]?://(?:[^\s()<>\"\']|(?:\([^\s()<>]*\)))+', msg):
                self.read_url(service, channel, url)

        return reply

    def default_prompt_prefix(self, service, channel):
        ''' The default prompt prefix '''
        ret = [
            f"It is {exact_time()} in the {natural_time()} on {today()}.",
            getattr(self.config.interact, "character", ""),
            f"{self.config.id.name} is feeling {self.recall.feels(self.recall.convo_id(service, channel))}.",
        ]
        goals = self.recall.list_goals(service, channel)
        if goals:
            ret.append(f"{self.config.id.name} is trying to accomplish the following goals: {', '.join(goals)}")
        else:
            log.warning(f"🙅‍♀️ No goal yet for {service} | {channel}")
        return '\n'.join(ret)

    def generate_prompt(self, summaries, convo, service, channel, lts=None):
        ''' Generate the model prompt '''
        newline = '\n'
        timediff = ''
        if lts and elapsed(lts, get_cur_ts()) > 600:
            timediff = f"It has been {ago(lts)} since they last spoke."

        # triples = set()
        graph_summary = ''
        convo_text = '\n'.join(convo)
        # for noun in self.extract_entities(convo_text) + self.extract_nouns(convo_text):
        #     for triple in self.recall.shortest_path(self.recall.bot_name, noun, src_type='Person'):
        #         triples.add(triple)
        # if triples:
        #     graph_summary = self.completion.model.triples_to_text(list(triples))

        # Is this just too much to think about?
        if self.completion.toklen(convo_text + newline.join(summaries)) > self.completion.max_prompt_length():
            log.warning("🥱 generate_prompt(): prompt too long, truncating.")
            convo_text = self.enc.decode(self.enc.encode(convo_text)[:self.completion.max_prompt_length()])

        return f"""{self.default_prompt_prefix(service, channel)}
{newline.join(summaries)}
{graph_summary}
{convo_text}
{timediff}
{self.config.id.name}:"""

    def get_status(self, service, channel):
        ''' status report '''
        return self.generate_prompt(
            self.recall.summaries(service, channel, size=3),
            self.recall.convo(service, channel, feels=True),
            service,
            channel
        )

    def extract_nouns(self, text):
        ''' return a list of all nouns (except pronouns) in text '''
        doc = self.completion.nlp(text)
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
        doc = self.completion.nlp(text)
        return list({n.text.strip() for n in doc.ents if n.text.strip() != self.config.id.name})

    def inject_idea(self, service, channel, idea, verb="recalls"):
        '''
        Directly inject an idea into recall memory.
        '''
        if verb != "decides" and idea in '\n'.join(self.recall.convo(service, channel, feels=True)):
            log.warning("🤌  Already had this idea, skipping:", idea)
            return

        self.recall.save_convo_line(
            service,
            channel,
            msg=idea,
            speaker_name=self.config.id.name,
            speaker_id=self.config.id.guid,
            convo_id=self.recall.convo_id(service, channel),
            verb=verb
        )

        log.warning(f"🤔 {verb}:", idea)
        return

    def surmise(self, service, channel, topic, size=10):
        ''' Stub for recall '''
        return self.recall.surmise(service, channel, topic, size)

    def add_goal(self, service, channel, goal):
        ''' Stub for recall '''
        return self.recall.add_goal(service, channel, goal)

    def get_goals(self, service, channel, goal=None, size=10):
        ''' Stub for recall '''
        return self.recall.get_goals(service, channel, goal, size)

    def list_goals(self, service, channel, size=10):
        ''' Stub for recall '''
        return self.recall.list_goals(service, channel, size)

    def read_news(self, service, channel, url, title):
        ''' Let's check the news while we ride the autobus. '''
        req = {
            "service": service,
            "channel": channel,
            "url": url,
            "title": title
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/read_news/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /read_news/ to interact: {err}")

    def read_url(self, service, channel, url):
        ''' Let's ride the autobus on the information superhighway. '''
        parsed = urlparse(url)
        if not bool(parsed.scheme) and bool(parsed.netloc):
            log.warning("👨‍💻 Not a URL:", url)
            return

        req = {
            "service": service,
            "channel": channel,
            "url": url
        }
        try:
            reply = requests.post(f"{self.config.interact.url}/read_url/", params=req, timeout=10)
            reply.raise_for_status()
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
            log.critical(f"🤖 Could not post /read_url/ to interact: {err}")
