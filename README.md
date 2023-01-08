# persyn
Personality Syndrome. Persyn for short.

Persyn makes it easy to integrate cutting-edge machine learning research projects into a single microservices framework. Each project implements a component of a greater personality (language model, long-term memory, visualizations, logic) that is tied together into an interactive group chat interface.

The resulting gestalt can be executed on any combination of local compute resources (CPU + GPU) and third-party APIs (OpenAI, Wikipedia, etc.)

The name "Personality Syndrome" was chosen by the first proto-Persyn instance itself, @AnnaTheBot@mas.to.

Major features:

  * Chat with any large language model over Slack, Discord, or Mastodon
  * Maintain a consistent and arbitrarily long train of thought
  * Short-term and long-term memory
  * Opinions
  * Auto-wikipedia
  * Auto-summarization of previous conversations over time or as the prompt size grows too long
  * Generate images on demand
  * Identify images dropped into the chat
  * Optional automatic image enhancement with Prompt Parrot


# Installation

Persyn in intended to run from inside a python virtualenv:

```
$ virtualenv --python=python3.8 env
$ . env/bin/activate
(env) $ pip install --upgrade pip # best practice
(env) $ pip install persyn
```

The default install only includes chat support.

If you'd also like to generate and post images:

```
(env) $ pip install persyn[all]
```

# Running a persyn with launch_bot

TBD

