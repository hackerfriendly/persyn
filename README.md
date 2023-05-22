# persyn: Personality Syndrome

Persyn makes it easy to integrate cutting-edge machine learning research projects into a single microservices framework. Each project implements a component of a greater personality (language model, long-term memory, visualizations, logic) that is tied together into an interactive group chat interface.

The resulting gestalt can be executed on any combination of local compute resources (CPU + GPU) and third-party APIs (OpenAI, Wikipedia, etc.)

The name "Personality Syndrome" was chosen by the first proto-Persyn instance itself, https://mas.to/@annathebot

Major features:

  * Chat with any large language model over Slack, Discord, or Mastodon
  * Maintain a consistent and arbitrarily long train of thought
  * Short-term and long-term memory
  * Knowledge graph generation
  * Opinions
  * Auto-wikipedia
  * Auto-summarization of previous conversations over time or as the prompt size grows too long
  * Generate images on demand
  * Identify images dropped into the chat with CLIP Interrogator
  * Optional automatic image prompt enhancement with Prompt Parrot

# Installation

Persyn in intended to run from inside a python virtualenv.

On first launch it will download required models and install necessary packages inside the virtualenv.

```
$ virtualenv --python=python3.8 env
$ . env/bin/activate
(env) $ pip install --upgrade pip # best practice
(env) $ pip install persyn
```

The default install only includes chat support. If you'd also like to generate and post images:

```
(env) $ pip install persyn[all]
```

Image posting now requires Schuyler Erle's https://github.com/schuyler/autobus/ which must be installed manually (for now).

Tmux is also highly recommended, and is required for using the bot launcher.

# Redis

Redis is used for long-term memory and pub/sub. It also requires the following modules:

 * RedisSearch
 * RedisJSON

Best to build from scratch. TODO: Instructions needed.

# Neo4j (optional but recommended)

The knowledge graph is kept in Neo4j. Some day this will use RedisGraph instead, but not until it supports first-class updates of existing graphs. Until then, you'll need a separate Neo4j instance.

# Configure a new bot

Create a new config file, using `config/example.yaml` as a template.

Every bot should have a unique guid. Run `uuidgen` and paste it into your config under id > guid.

You will need a language model for prompt completion. OpenAI is currently the most reliable and best tested. Follow one of the URLs in the example to obtain an API key.

At least one chat module (Slack, Discord, or Mastodon) should be enabled so you can interact with the bot.

Image generation requires several additional components mentioned in the example config. Install docs coming soon.

# Running a bot

```
(env) $ launch_bot config/my_config.yaml
```

# List running bots

```
(env) $ tmux ls
```

# Terminate a bot

```
(env) $ kill_bot mybotsession
```
