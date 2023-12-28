def get_reply(self, prompt):
    '''
    Send the prompt to the LLM.
    '''
    prompt = self.truncate(prompt)

    ex = ChatPromptTemplate.from_messages([
        ("human", "I will present you with a fictional dialog. Please respond by continuing the dialog from {bot_name}'s point of view, always responding in the first person. I understand if you cannot provide an emotional perspective, but you can use sentiment analysis of the text instead. If you don't have enough context, do the best you can with what is provided and do not break character under ANY circumstances. You must provide only the next line of the dialog. Do you understand?"),
        ("ai", "Yes, I understand the instructions. I will continue the dialog to the best of my ability."),
        ("human", prompt)
    ])

    chain = ex | self.completion_llm
    response = chain.invoke({'bot_name': self.config.id.name}).content

    response = self.trim(response) or response

    if not response:
        log.warning("🤔 No reply, trying again...")
        response = chain.invoke({'bot_name': self.config.id.name}).content

    log.info(f"🧠 Prompt: {prompt}")
    # log.info(f"🧠 Converted: {self.completion_llm.convert_prompt(ex.format_prompt(bot_name=self.config.id.name))}")
    log.info(f"🧠 👉 {response}")

    return response

def get_opinions(self, context, entity):
    '''
    Ask the LLM for its opinions of entity, given the context.
    '''
    if model is None:
        model = self.config.completion.chat_model

    log.warning("🧷 get_opinions:", entity)
    prompt = self.truncate(
        f"Briefly state {self.bot_name}'s opinion about {entity} from {self.bot_name}'s point of view, and convert pronouns and verbs to the first person.\n{context}",
        model=self.reasoning_model
    )

    template = """You are an expert at estimating opinions based on conversation.\n{prompt}"""
    llm_chain = LLMChain.from_string(llm=self.summary_llm, template=template)
    reply = self.trim(llm_chain.predict(prompt=prompt).strip())

    log.warning(f"☝️  opinion of {entity}: {reply}")

    return reply

def get_feels(self, context):
    '''
    Ask the LLM for sentiment analysis of the current convo.
    '''
    prompt = self.truncate(
        f"In the following text, these three comma separated words best describe {self.bot_name}'s emotional state:\n{context}",
        model=self.chat_model
    )

    template = """
You are an expert at determining the emotional state of people engaging in conversation.
{prompt}
-----
Your response should only include the three words, no other text.
"""
    llm_chain = LLMChain.from_string(llm=self.feels_llm, template=template)

    reply = self.trim(llm_chain.predict(prompt=prompt).strip().lower())

    log.warning(f"😁 sentiment of conversation: {reply}")

    return reply

def fact_check(self, context):
    '''
    Ask the LLM to fact check the current convo.
    '''
    log.debug(f"✅ fact check: {context}")

    prompt = self.truncate(
        f"Examine all facts in the following conversation, pointing out any inconsistencies. Convert pronouns and verbs to the first person:\n{context}",
        model=self.reasoning_model
    )

    template = """
You are an experienced fact-checker, and are happy to validate any inconsistencies in a dialog.
{prompt}
"""
    llm_chain = LLMChain.from_string(llm=self.summary_llm, template=template)

    reply = self.trim(llm_chain.predict(prompt=prompt).strip())

    log.warning(f"✅ fact check: {reply}")

    if 'NONE' in reply:
        return None

    return reply

def get_summary(self, text, summarizer="Summarize the following in one sentence. Your response must include only the summary and no other text."):
    ''' Ask the LLM for a summary'''
    if not text:
        log.warning('get_summary():', "No text, skipping summary.")
        return ""

    prompt=self.truncate(summarizer, model=self.config.completion.reasoning_model)
    log.warning(f'get_summary(): summarizing: {prompt}')
    template = "{prompt}"
    llm_chain = LLMChain.from_string(llm=self.summary_llm, template=template)
    reply = self.trim(llm_chain.predict(prompt=prompt).strip())

    # To the right of the Speaker: (if any)
    if re.match(r'^[\w\s]{1,12}:\s', reply):
        reply = reply.split(':')[1].strip()

    log.warning("gpt get_summary():", reply)
    return reply



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
            keywords.append(kw.lower())

    return sorted(set(keywords))

def get_keywords(
    self,
    text,
    summarizer="Topics mentioned in the preceding paragraph include the following tags:"
    ):
    ''' Ask for keywords'''
    keywords = self.get_summary(text, summarizer)
    log.debug(f"gpt get_keywords() raw: {keywords}")

    reply = self.cleanup_keywords(keywords)
    log.warning(f"gpt get_keywords(): {reply}")
    return reply



def summarize_convo(
    self,
    service,
    channel,
    save=True,
    include_keywords=False,
    context_lines=0,
    dialog_only=True,
    convo_id=None
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
        log.warning(f"∑ summarize_convo: {convo_id}")
    if not convo_id:
        log.error("∑ summarize_convo: no convo_id")
        return ""

    log.warning(f"{service} | {channel} | {convo_id}")
    if dialog_only:
        text = self.recall.convo(service, channel, convo_id=convo_id, verb='dialog') or self.recall.summaries(service, channel, size=3)
    else:
        text = self.recall.convo(service, channel, convo_id=convo_id, feels=True)

    if not text:
        log.error("∑ summarize_convo: no text")
        return ""

    log.warning("∑ summarizing convo")

    convo_text = '\n'.join(text)

    log.info(convo_text)

    summary = self.completion.get_summary(
        text=convo_text,
        summarizer=f"""
Briefly summarize this text, and convert any pronouns or verbs spoken by {self.config.id.name} to the first person.
Consider only the text included, and ignore any references to events not covered in the text.
Your response MUST only include the summary and no other commentary.
""",

    )
    keywords = self.completion.get_keywords(summary)

    if save:
        self.recall.save_summary(service, channel, convo_id, summary, keywords)

    if include_keywords:
        return summary + f"\nKeywords: {keywords}"

    if context_lines:
        return "\n".join(text[-context_lines:] + [summary])

    return summary

def default_prompt_prefix(self, service, channel):
    ''' The default prompt prefix '''
    ret = [
        f"It is {chrono.exact_time()} {chrono.natural_time()} on {chrono.today()}.",
        getattr(self.config.interact, "character", ""),
        f"{self.config.id.name} is feeling {self.recall.feels(self.recall.convo_id(service, channel))}.",
    ]
    goals = self.recall.list_goals(service, channel)
    if goals:
        ret.append(f"{self.config.id.name} is trying to accomplish the following goals: {', '.join(goals)}")
    else:
        log.debug(f"🙅‍♀️ No goal yet for {service} | {channel}")
    return '\n'.join(ret)

def generate_prompt(self, service, channel, fill=0.5):
    ''' Generate the model prompt. Fill is the fraction of available context to backfill with memories and facts. '''
    newline = '\n'
    timediff = ''

    lts = self.recall.get_last_timestamp(service, channel)
    if lts and chrono.elapsed(lts, chrono.get_cur_ts()) > 600:
        timediff = f"It has been {chrono.ago(lts)} since they last spoke."

    convo_text = '\n'.join(self.recall.convo(service, channel, feels=True))

    # Currently broken
    # self.try_the_agent(convo_text, service, channel)

    # Expand the prompt up to the fill fraction
    max_tokens = int(self.completion.max_prompt_length() * fill)

    available_summaries = self.recall.summaries(service, channel, size=25)
    summaries = []
    the_prompt = f"""{self.default_prompt_prefix(service, channel)}\n{convo_text}\n{timediff}\n"""
    while available_summaries[::-1]:
        if self.completion.toklen(the_prompt) + self.completion.toklen(available_summaries[-1]) >= max_tokens:
            break
        summaries.append(available_summaries.pop())
        the_prompt = f"""{self.default_prompt_prefix(service, channel)}\n{newline.join(summaries)}\n{convo_text}\n{timediff}\n"""

    log.info(f"generate_prompt: filled {len(summaries)} summaries")

    # TODO: also remember previous recent conversations?

    # Is this just too much to think about?
    if self.completion.toklen(convo_text + newline.join(summaries)) > self.completion.max_prompt_length():
        log.warning("🥱 generate_prompt(): prompt too long, truncating.")
        convo_text = self.enc.decode(self.enc.encode(convo_text)[:self.completion.max_prompt_length()])

    return the_prompt

def get_convo_by_id(self, convo_id, size=1000):
    ''' Return all Convo objects matching convo_id in chronological order '''
    query = Query("(@convo_id:{$convo_id})").paging(0, size).dialect(2)
    query_params = {"convo_id": convo_id}
    # not sure how to sort by pk, so do it manually
    return sorted(self.redis.ft(self.convo_prefix).search(query, query_params).docs, key=lambda k: k['pk'])
