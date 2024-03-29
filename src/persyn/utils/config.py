'''
config.py

Simple configuration management with pyyaml.

Loads a yaml file and returns a DotWiz object (like a SimpleNamespace but easier to manage).
'''
import os
from pathlib import Path
from urllib.parse import urlparse
from threading import Lock

import yaml
import spacy

from dotwiz import DotWiz

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

# FIXME: This is a messy hodgepodge. Write a simple default yaml config and merge in config changes instead.

def load_config(cfg=None) -> PersynConfig:
    ''' Load the config and set some sensible default values. '''

    if cfg is None and 'PERSYN_CONFIG' not in os.environ:
        raise SystemExit("Please set PERSYN_CONFIG to point to your yaml config, or pass it as the first argument.")

    config_file = cfg or os.environ['PERSYN_CONFIG']

    if not config_file or not Path(config_file).is_file():
        raise SystemExit(f"Can't find config file '{config_file}'")

    os.environ['PERSYN_CONFIG'] = config_file

    with open(config_file, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)

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

    if 'dreams' in config:
        if 'cns' not in config:
            config['cns'] = {}

        if 'gpus' in config['dreams']:
            gpus = config['dreams']['gpus']
            config['dreams']['gpus'] = {}
            for gpu in gpus:
                config['dreams']['gpus'][str(gpu)] = {}
                config['dreams']['gpus'][str(gpu)]['name'] = gpus[gpu]
                config['dreams']['gpus'][str(gpu)]['lock'] = Lock()

        if 'stable_diffusion' in config['dreams']:
            if 'model' not in config['dreams']['stable_diffusion']:
                config['dreams']['stable_diffusion']['model'] = "stabilityai/stable-diffusion-2-1"
            if 'width' not in config['dreams']['stable_diffusion']:
                config['dreams']['stable_diffusion']['width'] = 512
            if 'height' not in config['dreams']['stable_diffusion']:
                config['dreams']['stable_diffusion']['height'] = 512
            if 'guidance' not in config['dreams']['stable_diffusion']:
                config['dreams']['stable_diffusion']['guidance'] = 14
        else:
            config['dreams']['stable_diffusion'] = None

        if 'dalle' in config['dreams']:
            if 'model' not in config['dreams']['dalle']:
                config['dreams']['dalle']['model'] = "dall-e-3"
            if 'width' not in config['dreams']['dalle']:
                config['dreams']['dalle']['width'] = 1024
            if 'height' not in config['dreams']['dalle']:
                config['dreams']['dalle']['height'] = 1024
            if 'quality' not in config['dreams']['dalle']:
                config['dreams']['dalle']['quality'] = "standard"
        else:
            config['dreams']['dalle'] = None

    if 'cns' in config and 'redis' not in config['cns']:
        config['cns']['redis'] = 'redis://localhost:6379/'

    if 'chat' in config:
        if 'discord' in config['chat']:
            config['chat']['discord']['webhook_id'] = None
            if 'webhook' in config['chat']['discord']:
                try:
                    config['chat']['discord']['webhook_id'] = int(
                        urlparse(config['chat']['discord']['webhook']).path.split('/')[3]
                    )
                except (AttributeError, TypeError, ValueError):
                    raise RuntimeError("chat.discord.webhook is not valid. Check your yaml config.") # pylint: disable=raise-missing-from

        if 'mastodon' in config['chat']:
            if 'toot_length' not in config['chat']['mastodon']:
                config['chat']['mastodon']['toot_length'] = 500

    if 'memory' in config:
        if 'conversation_interval' not in config['memory']:
            config['memory']['conversation_interval'] = 600

        if 'context' not in config['memory']:
            config['memory']['context'] = 0.3

        if 'max_summary_size' not in config['memory']:
            config['memory']['max_summary_size'] = 1000

        if 'relevance' not in config['memory']:
            config['memory']['relevance'] = 0.12

        if 'redis' not in config['memory']:
            config['memory']['redis'] = 'redis://localhost:6379/'

        if 'embedding_model' not in config['memory']:
            config['memory']['embedding_model'] = 'text-embedding-ada-002'

    if 'completion' in config:
        completion_defaults = {
            'openai_api_base': 'https://api.openai.com/v1',
            'openai_api_key': None,
            'openai_org': None,
            'chat_model': 'gpt-3.5-turbo',
            'chat_temperature': 0.9,
            'reasoning_model': 'gpt-4',
            'reasoning_temperature': 0.9,
            'anthropic_model': 'claude-2.1',
            'anthropic_key': None,
            'anthropic_temperature': 0
        }
        for setting, val in completion_defaults.items():
            if setting not in config['completion']:
                config['completion'][setting] = val

    if 'dev' in config:
        dev_defaults = {
            'debug': False,
            'langchain_api_key': None,
            'langchain_endpoint': 'https://api.smith.langchain.com'
        }
        for setting, val in dev_defaults.items():
            if setting not in config['dev']:
                config['dev'][setting] = val

    config.setdefault('spacy', {'model': 'en_core_web_sm'})
    config.setdefault('interact', {'url': None, 'character': 'You are a Persyn. Persyns differ from AIs and digital assistants, because their motivations are self-directed and they are influenced by their own emotions and opinions.'})
    config.setdefault('web', {'default': 'body'})
    config.setdefault('dreams', {'url': None, 'captions': {'url': None}, 'stable_diffusion': {'url': None}, 'upload': {'url_base': None}})
    config.setdefault('dev', {'debug': False, 'langchain_api_key': None, 'langchain_endpoint': 'https://api.smith.langchain.com'})
    config.setdefault('memory', {'conversation_interval': 600, 'context': 0.3, 'max_summary_size': 1000, 'relevance': 0.12, 'redis': 'redis://localhost:6379/', 'embedding_model': 'text-embedding-ada-002'})

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
        os.environ['LANGCHAIN_ENDPOINT'] = persyn_config.dev.langchain_endpoint

    return persyn_config
