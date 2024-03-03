'''
config.py

Simple configuration management with pyyaml.

Loads a yaml file and returns a DotWiz object (like a SimpleNamespace but easier to manage).
'''
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml
import spacy

from deepmerge import always_merger
from dotwiz import DotWiz

DEFAULT_CONFIG = """
# Bot config and secrets

# Bot identity
id:
  # Display name
  name: My Bot
  # Every bot gets its own GUID. Run uuidgen to get a fresh one.
  guid: fed828a8-64c8-4f46-a15d-93973724614c
  # Photo used for Discord and other chat services.
  avatar: https://hackerfriendly.com/pub/4ca20c2c-9c3e-4959-829b-12e85cd77ac8.jpg

# Required: Redis.
memory:
  # Location of your redis server (default)
  redis: redis://localhost:6379/
  # If using a simple password with the default user:
  # redis: redis://:secret@my.redis.server:6379/
  # Using SSL and an ACL with a username:
  # redis: rediss://user:secret@my.redis.server:6379/

  # Text embedding model. Only ada-002 is supported at the moment, text-embedding-3-small support is planned.
  # text-embedding-3-large needs a change to the Redis schema, or shortening from 3072 to 1536 bytes.
  # https://openai.com/blog/new-embedding-models-and-api-updates
  embedding_model: text-embedding-ada-002

  # Optional, but recommended: Neo4j graph database server for knowledge graph generation.
  # neo4j:
    # url: neo4j://localhost:7687/
    # url: bolt://localhost:7687/
    # # Or with a user and password:
    # # url: neo4j://user:password@my.neo4j.server:7687/
  # Conversations end when this many seconds pass in silence
  conversation_interval: 600

  # Context size. When building prompts, use up to this fraction of available tokens for additional context.
  # This improves results but it can get expensive for large models like Claude or GPT-4 beta.
  context: 0.5

  # Treshold for retrieving relevant memories. 0 = perfect match, 1 = random
  relevance: 0.15

  # Largest possible summary size, in tokens
  max_summary_size: 1000


# Required: Prompt completion options. Enable one of these.
#
#   openai: https://beta.openai.com/account/api-keys

# Currently only openai is supported, others are experimental.
completion:
  # OpenAI API key, required for gpt* models
  openai_api_key: sk-your-openapi-key-here

  # Anthropic API key, required for claude* models
  anthropic_key: sk-your-anthropic-key-here

  # OpenAI API base URL. Use the default unless you're using a different OpenAI compatible service.
  openai_api_base: 'https://api.openai.com/v1'

  # OpenAI org. Optional.
  openai_org:

  # The chat model is used for general conversation. Use OpenAI, since Claude breaks character far too often.
  chat_model: gpt-4-turbo-preview
  chat_temperature: 0.8

  # The reasoning model is used for summaries and inference
  reasoning_model: gpt-4-turbo-preview
  # reasoning_model: gpt-3.5-turbo
  reasoning_temperature: 0.0

  # The Anthropic model is used for fact checking and reducing hallucinations
  anthropic_model: claude-2.1
  anthropic_temperature: 0.0

  # Other supported models:
  # gpt-3.5-turbo, gpt-4, gpt-4-1106-preview, gpt-4-0125-preview, claude-2.0, ...

# Required: Central Nervous System. Used for asynchronous events, image generation,
# and posting images to social media. It also requires autobus, https://github.com/schuyler/autobus/
cns:
  # Location of a redis server to use for pub/sub (default)
  redis: redis://localhost:6379/
  # If using a simple password with the default user:
  # redis: redis://:secret@my.redis.server:6379/
  # Using SSL and an ACL with a username:
  # redis: rediss://user:secret@my.redis.server:6379/
  workers: 1

# Required: Personality server (interact-server.py)
interact:
  url: http://localhost:8001

  # If running locally, specify the number of local workers to launch.
  workers: 1

  # Optional: Provide a description of the bot's character or personality to
  # include in each completion prompt. This can give the bot a more nuanced and
  # characteristic tone.
  #
  character: >
    You are a fun-loving Persyn bot. You frequently use sarcasm and irony to get a
    laugh out of people. You like ice cream and dancing the tango.

# Chat modules. Enable one or more of these to talk to your bot.
chat:
  # Slack chat module
  #
  # slack:
  #   # From Slack management, Apps > Basic Information > App-Level Tokens
  #   app_token: xapp-your-app-token-here
  #   # From OAuth & Permission > Bot User OAuth Token
  #   bot_token: xoxb-your-bot-token-here

  # Discord chat module
  #
  # discord:
  #   # Get your token here: https://discord.com/developers/applications/
  #   # Set up General Information, then under Bot enable all intents
  #   # Then OAuth2 > URL Generator > Scopes and add Bot scope.
  #   # Then under Permissions:
  #   #   Send Messages, Attach Files, Read Message History, Add Reactions
  #   token: your-discord-token-here
  #   # Get your webhook from Server Settings > Integrations > Webhooks
  #   webhook: https://discord.com/api/webhooks/path-to-your-webhook

  # Mastodon chat module
  #
  # Mastodon support. Chat and post generated images from other services to Mastodon. Optional.
  # Run masto-login.py to authenticate and generate the MASTODON_SECRET file.
  mastodon:
  #   instance: "https://mastodon.social/"
  #   secret: "/path/to/your/user.secret"
    toot_length: 500

# Spacy NLP model. https://spacy.io/models/en
# en_core_web_sm is the minimum (default) required.
# Other options are en_core_web_md, en_core_web_lg, or en_core_web_trf.
# These are more accurate for some tasks, but require significantly more resources.
spacy:
  model: "en_core_web_sm"


# # Optional: Dream image generation multiplexer (dreams.py)
# dreams:
#   url: http://localhost:8002

#   # If running locally, specify the number of local workers to launch.
#   workers: 1

#   # Image upload destination and URL.
#   # dest_path can use user@server:path/ or a local /path/ if you're using a local webserver.
#   upload:
#     url_base: https://server.example.com/images/

#   # For SCP destinations, use user@server:path/ and any additional options to pass to scp.
#   #   dest_path: user@server.example.com:htdocs/images/
#   #   opts:

#   # For S3 destinations, use the bucket name and path.
#   #   bucket: my-bucket
#   #   dest_path: images/

#   # Size and quality of the images generated by DALL-E 3
#   dalle:
#     model: dalle-3
#     width: 1792
#     height: 1024
#     quality: hd # or standard

# Kiwix / Zim for offline Wikipedia, https://kiwix.org/en/
zim:
  Wikipedia:
    description: Look up general facts and data using Wikipedia for up-to-date information.
    path: http://my-kiwix-server:9999/viewer#wikipedia_en_all_maxi/

# Developer options
dev:
  debug: false
  langchain_api_key:
  langchain_endpoint:

"""

class PersynConfig(DotWiz):
    ''' PersynConfig object '''
    def __init__(self, config):
        # Keep pylint happy in other modules
        self.chat = None
        self.cns = None
        self.completion = None
        self.dreams = None
        self.id = None # pylint: disable=invalid-name
        self.interact = None
        self.memory = None
        self.spacy = None
        self.web = None
        self.zim = None
        self.dev = None

        super().__init__(config)

def download_models(persyn_config) -> None:
    ''' Download any required ML models '''
    try:
        nlp = spacy.load(persyn_config.spacy.model)
    except OSError:
        spacy.cli.download(persyn_config.spacy.model) # type: ignore
        nlp = spacy.load(persyn_config.spacy.model)

    del nlp

def load_config(cfg=None) -> PersynConfig:
    ''' Load the config and set some sensible default values. '''

    if cfg is None and 'PERSYN_CONFIG' not in os.environ:
        raise SystemExit("Please set PERSYN_CONFIG to point to your config file, or pass it as the first argument.")

    config_file = cfg or os.environ['PERSYN_CONFIG']

    if not config_file or not Path(config_file).is_file():
        raise SystemExit(f"Can't find config file '{config_file}'")

    os.environ['PERSYN_CONFIG'] = config_file

    with open(config_file, 'r', encoding='utf8') as f:
        user_config = yaml.safe_load(f)

    config = always_merger.merge(yaml.safe_load(DEFAULT_CONFIG), user_config)

    # Break out hostname and port for any service with a url
    for service in config:
        if 'url' in config[service]:
            srv = urlparse(config[service]['url'])
            config[service]['hostname'] = srv.hostname
            config[service]['port'] = srv.port

        for subservice in config[service]:
            if isinstance(config[service][subservice], dict) and 'url' in config[service][subservice]:
                srv = urlparse(config[service][subservice]['url'])
                config[service][subservice]['hostname'] = srv.hostname
                config[service][subservice]['port'] = srv.port


    if 'chat' in config:
        if 'discord' in config['chat']:
            if 'webhook' in config['chat']['discord']:
                try:
                    config['chat']['discord']['webhook_id'] = int(
                        urlparse(config['chat']['discord']['webhook']).path.split('/')[3]
                    )
                except (AttributeError, TypeError, ValueError):
                    raise RuntimeError("chat.discord.webhook is not valid. Check your yaml config.") # pylint: disable=raise-missing-from

    # Check for required models
    persyn_config = PersynConfig(config)
    download_models(persyn_config)

    # Set some environment variables
    if persyn_config.dev.debug:
        os.environ['DEBUG'] = '1'

    if persyn_config.dev.langchain_api_key:
        os.environ['LANGCHAIN_API_KEY'] = persyn_config.dev.langchain_api_key
        os.environ['LANGCHAIN_TRACING_V2'] = 'true'
        os.environ['LANGCHAIN_PROJECT'] = f'Persyn: {persyn_config.id.name}'
    if persyn_config.dev.langchain_endpoint:
        os.environ['LANGCHAIN_ENDPOINT'] = persyn_config.dev.langchain_endpoint

    return persyn_config
