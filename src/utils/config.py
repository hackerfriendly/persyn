'''
config.py

Simple configuration management with pyyaml.

Loads a yaml file and returns a SimpleNamespace.
'''
import os
from pathlib import Path
from urllib.parse import urlparse
from threading import Lock
from subprocess import run

import yaml
import spacy
import coreferee

from dotwiz import DotWiz # pylint: disable=no-member

def download_models(persyn_config):
    ''' Download any required ML models '''
    try:
        nlp = spacy.load(persyn_config.spacy.model)
    except OSError:
        spacy.cli.download(persyn_config.spacy.model)
        nlp = spacy.load(persyn_config.spacy.model)

    try:
        nlp.add_pipe('coreferee')
    except coreferee.errors.ModelNotSupportedError:
        run(['python', '-m', 'coreferee', 'install', 'en'], shell=False, check=True)
        nlp.add_pipe('coreferee')

    nlp.remove_pipe('coreferee')
    del nlp

def load_config(cfg=None):
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
        if 'engines' in config['dreams']:
            config['dreams']['all_engines'] = list(config['dreams']['engines'].keys())

            for engine in config['dreams']['all_engines']:
                if not config['dreams']['engines'][engine]:
                    config['dreams']['engines'][engine] = {}
                    config['dreams']['engines'][engine]['models'] = ["default"]

        if 'gpus' in config['dreams']:
            gpus = config['dreams']['gpus']
            config['dreams']['gpus'] = {}
            for gpu in gpus:
                config['dreams']['gpus'][str(gpu)] = {}
                config['dreams']['gpus'][str(gpu)]['name'] = gpus[gpu]
                config['dreams']['gpus'][str(gpu)]['lock'] = Lock()

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

        if 'elastic' in config['memory']:
            elastic_defaults = {
                'version': 'v0',
                'timeout': 30,
                'index_prefix': config['id']['name'].replace(' ', '').lower()
            }

            for setting, val in elastic_defaults.items():
                if setting not in config['memory']['elastic']:
                    config['memory']['elastic'][setting] = val

    config.setdefault('spacy', {'model': 'en_core_web_sm'})
    config.setdefault('sentiment', {})

    # Check for required models
    persyn_config = DotWiz(config)
    download_models(persyn_config)

    return persyn_config
