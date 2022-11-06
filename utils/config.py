'''
config.py

Simple configuration management with pyyaml.

Loads a yaml file and returns a SimpleNamespace.
'''
import os
from pathlib import Path
# from urllib.parse import urlparse

import yaml
from dotwiz import DotWiz

if 'PERSYN_CONFIG' not in os.environ:
    raise SystemExit(f"Please set PERSYN_CONFIG to point to your yaml config.")

# TODO: semantic sanity checking
def load_config():
    config_file = os.environ['PERSYN_CONFIG']

    if not Path(config_file).is_file():
        raise SystemExit(f"Can't find config file '{config_file}'")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if 'dreams' in config and 'engines' in config['dreams']:
        config['dreams']['all_engines'] = list(config['dreams']['engines'].keys())

    for engine in config['dreams']['all_engines']:
        if not config['dreams']['engines'][engine]:
            config['dreams']['engines'][engine] = {}
            config['dreams']['engines'][engine]['models'] = ["default"]

    return DotWiz(config)
