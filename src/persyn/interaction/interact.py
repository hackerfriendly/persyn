'''
interact.py

The limbic system library.
'''
# pylint: disable=import-error, wrong-import-position, wrong-import-order
import random
import re # used by custom filters

from urllib.parse import urlparse

import requests

from langchain import LLMMathChain
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.tools import WikipediaQueryRun, PubmedQueryRun
# from langchain.tools.python.tool import PythonREPLTool # unsafe without a sandbox!
from langchain.utilities import WikipediaAPIWrapper

# Long and short term memory
from persyn.interaction.memory import Recall

# Time handling
from persyn.interaction.chrono import exact_time, natural_time, ago, today, elapsed, get_cur_ts

# Prompt completion
from persyn.interaction.completion import LanguageModel

# Color logging
from persyn.utils.color_logging import log

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

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

        self.llm = ChatOpenAI(model=persyn_config.completion.chat_model, temperature=0)

        llm_math_chain = LLMMathChain.from_llm(llm=self.llm, verbose=True)

        @tool
        def consult_wikipedia(query: str) -> str:
            """Check Wikipedia for facts"""
            reply = wikipedia.run(query)
            replylen = self.completion.toklen(reply)
            maxlen = round(self.completion.max_prompt_length() * 0.8)
            if replylen > maxlen:
                log.warning(f"wikipedia: reply too long ({replylen}), truncating to {maxlen}")

                enc = self.completion.model.get_enc()
                reply = enc.decode(enc.encode(reply)[:maxlen])

            return reply

        self.agent = initialize_agent(
            [
                take_a_photo,
                consult_wikipedia,
                PubmedQueryRun(),
                # PythonREPLTool(),
                # Tool(
                #     name="Python",
                #     func=PythonREPLTool.run,
                #     description="Run python programs to answer questions about math or programming",
                #     return_direct=True
                # ),
                Tool(
                    name="Calculator",
                    func=llm_math_chain.run,
                    description="Answer math questions accurately",
                    return_direct=True
                ),
                Tool(
                    name="Say something",
                    func=lambda x: f"say:{x}",
                    description="Continue the conversation",
                    return_direct=True
                ),
            ],
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )

    def summarize_convo(
        self,
        service,
        channel,
        save=True,
        max_tokens=200,
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
            summarizer=f"Briefly summarize this conversation from {self.config.id.name}'s point of view, and convert pronouns and verbs to the first person.",
            max_tokens=max_tokens,
            model=model
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

    def choose_response(self, prompt, convo, service, channel, goals, max_tokens=150):
        ''' Choose the best completion response from a list of possibilities '''
        if not convo:
            convo = []

        log.info("👍 Choosing a response with model:", self.config.completion.summary_model)
        scored = self.completion.get_replies(
            prompt=prompt,
            convo=convo,
            goals=goals,
            model=self.config.completion.summary_model,
            n=3,
            max_tokens=max_tokens
        )

        if not scored:
            log.warning("🤨 No surviving replies, try again with model:",
                        self.config.completion.chat_model or self.config.completion.completion_model)
            scored = self.completion.get_replies(
                prompt=prompt,
                convo=convo,
                goals=goals,
                model=self.config.completion.chat_model or self.config.completion.completion_model,
                n=2,
                max_tokens=max_tokens
            )

        # Uh-oh. Just ignore whatever was last said.
        if not scored:
            log.warning("😳 No surviving replies, one last try with model:", self.config.completion.completion_model)
            scored = self.completion.get_replies(
                prompt=self.generate_prompt([], convo[:-1], service, channel),
                convo=convo,
                goals=goals,
                max_tokens=max_tokens
            )

        if not scored:
            log.warning("😩 No surviving replies, I give up.")
            log.info("🤷‍♀️ Choice: none available")
            return (":shrug:", [])

        for item in sorted(scored.items()):
            log.warning(f"{item[0]:0.2f}:", item[1])

        idx = random.choices(list(sorted(scored)), weights=list(sorted(scored)))[0]
        reply = scored[idx]
        del scored[idx]
        log.info(f"✅ Choice: {idx:0.2f}", reply)

        return (reply, [item[1] for item in scored.items()])

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
    def get_reply(self, service, channel, msg, speaker_name, speaker_id, send_chat=True, max_tokens=150):  # pylint: disable=too-many-locals
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
        if convo and len(convo) % 5 == 0:
            self.save_knowledge_graph(service, channel, convo_id, convo)

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
        prompt = self.generate_prompt(summaries, convo, service, channel, lts)

        # Is this just too much to think about?
        if (self.completion.toklen(prompt) + max_tokens) > self.completion.max_prompt_length():
            # Kick off a summary request via autobus. Yes, we're talking to ourselves now.
            log.warning("🥱 get_reply(): prompt too long, summarizing.")
            req = {
                "service": service,
                "channel": channel,
                "save": True,
                "max_tokens": 100
            }
            try:
                reply = requests.post(f"{self.config.interact.url}/summary/", params=req, timeout=60)
                reply.raise_for_status()
            except (requests.exceptions.RequestException, requests.exceptions.ConnectionError) as err:
                log.critical(f"🤖 Could not post /summary/ to interact: {err}")
                return " :dancer: :interrobang: "

            prompt = self.generate_prompt([], convo, service, channel, lts)

        ret = self.agent.run(prompt)
        log.warning("🕵️‍♂️ ", ret)

        others = []
        if ret.startswith("say:"):
            log.warning("🗣️ Say something about:", ret)
            self.inject_idea(service, channel, ret.split(':', maxsplit=1)[1].strip())
            convo = self.recall.convo(service, channel, feels=True)
            prompt = self.generate_prompt(summaries, convo, service, channel, lts)
            (reply, others) = self.choose_response(prompt, convo, service, channel, self.recall.list_goals(service, channel), max_tokens)

        elif ret.startswith("photo:"):
            prompt = ret.split(':', maxsplit=1)[1].strip()
            log.warning("Take a photo of:", prompt)
            self.generate_photo(service, channel, prompt)
            return ""

        else:
            log.warning("🌍 Wikipedia:", ret)
            self.inject_idea(service, channel, ret)
            convo = self.recall.convo(service, channel, feels=True)
            prompt = self.generate_prompt(summaries, convo, service, channel, lts)
            (reply, others) = self.choose_response(prompt, convo, service, channel, self.recall.list_goals(service, channel), max_tokens)


        if self.custom_filter:
            try:
                reply = self.custom_filter(reply)
            except Exception as err: # pylint: disable=broad-except
                log.warning(f"🤮 Custom filter failed: {err}")

        # Say it!
        if send_chat:
            self.send_chat(service, channel, reply)

        for idea in others:
            self.inject_idea(service, channel, idea, verb="thinks")

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
            f"It is {exact_time()} on {today()} ({natural_time()}).",
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
