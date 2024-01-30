'''
memory (redis) tests
'''
# pylint: disable=import-error, wrong-import-position, invalid-name, no-member

import pytest

from src.persyn.interaction.memory import Recall
from src.persyn.interaction.goals import Goal

# Bot config
from src.persyn.utils.config import load_config

# from utils.color_logging import log

persyn_config = load_config()

def clear_ns(ns, chunk_size=5000):
    ''' Clear a namespace '''
    recall = Recall(persyn_config)

    if not ns or ':' not in ns:
        return False

    cursor = '0'
    while cursor != 0:
        cursor, keys = recall.redis.scan(cursor=cursor, match=f"{ns}*", count=chunk_size)
        if keys:
            recall.redis.delete(*keys)
    return True

@pytest.fixture
def cleanup():
    ''' Delete everything with the test bot_id '''
    yield

    recall = Recall(persyn_config)
    clear_ns(recall.convo_prefix)
    clear_ns(recall.goal_prefix)

    for idx in [recall.convo_prefix, recall.opinion_prefix, recall.goal_prefix, recall.news_prefix]:
        try:
            recall.redis.ft(idx).dropindex()
        except recall.redis.exceptions.ResponseError as err:
            print(f"Couldn't drop index {idx}:", err)

@pytest.fixture
def goal(conversation_interval=2):
    return Goal(persyn_config)

def test_add(goal, cleanup):
    goal_id = goal.add("service", "channel", "goal")
    assert goal_id is not None

def test_add_action(goal, cleanup):
    goal_id = goal.add("service", "channel", "goal")
    goal.add_action(goal_id, "action")
    actions = goal.list_actions(goal_id)
    assert "action" in actions
    assert len(actions) == 1

    goal.add_action(goal_id, "action")
    actions = goal.list_actions(goal_id)
    assert len(actions) == 1 # duplicate actions not permitted

    goal.add_action(goal_id, "action2")
    actions = goal.list_actions(goal_id)
    assert len(actions) == 2

def test_delete_action(goal, cleanup):
    goal_id = goal.add("service", "channel", "goal")
    goal.add_action(goal_id, "action")
    goal.delete_action(goal_id, "action")
    actions = goal.list_actions(goal_id)
    assert "action" not in actions

def test_list_actions(goal, cleanup):
    goal_id = goal.add("service", "channel", "goal")
    goal.add_action(goal_id, "action")
    actions = goal.list_actions(goal_id)
    assert actions == ["action"]

def test_fetch_goal(goal, cleanup):
    goal_id = goal.add("service", "channel", "goal")
    goal = goal.fetch(goal_id)
    assert goal["content"] == "goal"
    assert goal["goal_id"] == goal_id

def test_list_for_channel(goal, cleanup):
    goal.add("service", "channel", "goal")
    goals = goal.list_for_channel("service", "channel")
    assert len(goals) == 1

    goal.add("service", "channel", "goal2")
    goals = goal.list_for_channel("service", "channel")
    assert len(goals) == 1 # default size == 1

    goals = goal.list_for_channel("service", "channel", size=10)
    assert len(goals) == 2

    goal.add("service", "channel", "goal2")
    goals = goal.list_for_channel("service", "channel", size=10)
    assert len(goals) == 3 # duplicate goals permitted

def test_achievement(goal, cleanup):
    goal_id = goal.add("service", "channel", "goal")
    assert goal.achieved(goal_id) == False

    goal.achieve(goal_id, 0.9)
    assert goal.achieved(goal_id) == False

    goal.achieve(goal_id, 0.1)
    assert goal.achieved(goal_id) == True

    goal.achieve(goal_id, -0.1)
    assert goal.achieved(goal_id) == False
