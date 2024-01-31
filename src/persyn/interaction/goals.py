''' goals.py: Goals and actions. '''

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import ulid

from redis.commands.search.query import Query

from persyn.interaction.memory import Recall, decode_dict
from persyn.utils.color_logging import ColorLog

log = ColorLog()

@dataclass
class Goal(Recall):
    ''' Goals and actions '''

    def add(self, service: str, channel: str, goal: str) -> str:
        ''' Add a new goal. Returns the goal_id. '''
        goal_id = str(ulid.ULID())

        self.redis.hset(f"{self.goal_prefix}:{goal_id}:meta", "service", service)
        self.redis.hset(f"{self.goal_prefix}:{goal_id}:meta", "channel", channel)
        self.redis.hset(f"{self.goal_prefix}:{goal_id}:meta", "goal_id", goal_id)
        self.redis.hset(f"{self.goal_prefix}:{goal_id}:meta", "content", goal)
        self.redis.hset(f"{self.goal_prefix}:{goal_id}:meta", "content_vector", self.lm.get_embedding(goal))
        self.redis.hset(f"{self.goal_prefix}:{goal_id}:meta", "achieved", 0)

        return goal_id

    def add_action(self, goal_id: str, action: str) -> None:
        ''' Add an action to a goal '''
        self.redis.sadd(f"{self.goal_prefix}:{goal_id}:actions", action)

    def delete_action(self, goal_id: str, action: str) -> None:
        ''' Delete an action from a goal '''
        self.redis.srem(f"{self.goal_prefix}:{goal_id}:actions", action)

    def list_actions(self, goal_id: str) -> List[str]:
        ''' List actions for a goal '''
        return [action.decode() for action in self.redis.smembers(f"{self.goal_prefix}:{goal_id}:actions")]

    def fetch(self, goal_id: str) -> Dict[str, Any]:
        ''' Fetch a goal from Redis '''
        return decode_dict(self.redis.hgetall(f"{self.goal_prefix}:{goal_id}:meta"))

    def list_for_channel(self, service: str, channel: str, size: Optional[int] = 1) -> List[Dict[str, Any]]:
        ''' List goals for a channel '''

        query = (
            Query("((@service:{$service}) (@channel:{$channel}))")
            .sort_by("achieved", asc=False)
            .return_fields("service", "channel", "goal_id", "content", "achieved")
            .paging(0, size)
            .dialect(2)
        )
        query_params = {"service": service, "channel": channel}

        return self.redis.ft(self.goal_prefix).search(query, query_params).docs

    def achieve(self, goal_id: str, amount: Optional[float] = 0.01) -> float:
        ''' Achieve a goal by a given amount. To reduce progress, pass a negative amount. '''
        return self.redis.hincrbyfloat(f"{self.goal_prefix}:{goal_id}:meta", "achieved", amount)

    def achieved(self, goal_id: str) -> bool:
        ''' Return True if the goal is achieved, else False '''
        return float(self.redis.hget(f"{self.goal_prefix}:{goal_id}:meta", "achieved")) >= 1.0

    def undertake_action(self, convo_id: str, goal_id: str, action: str) -> None:
        '''
        Save a goal + action for later evaluation when the conversation has expired.
        '''
        self.redis.hset(f"{self.convo_prefix}:{convo_id}:actions", goal_id, action)

    def list_undertaken_actions(self, convo_id: str) -> List[Dict[str, Any]]:
        '''
        List actions undertaken in a conversation.
        '''
        return decode_dict(self.redis.hgetall(f"{self.convo_prefix}:{convo_id}:actions"))

    def find_related_goals(
        self,
        service: str,
        channel: str,
        text: str,
        threshold: Optional[float] = 1.0,
        size: Optional[int] = 1
        ) -> List:
        '''
        Find goals related to text using vector similarity
        '''
        log.debug(f"find_related_goals: {service} {channel} '{text}' {threshold} {size}")

        emb = self.lm.get_embedding(text)
        query = (
            Query(
                "((@service:{$service}) (@channel:{$channel})) @content_vector:[VECTOR_RANGE $threshold $emb]=>{$YIELD_DISTANCE_AS: score}"
            )
            .sort_by("score")
            .return_fields("service", "channel", "goal_id", "content", "score", "achieved")
            .paging(0, size)
            .dialect(2)
        )
        reply = self.redis.ft(self.goal_prefix).search(query, {"service": service, "channel": channel, "emb": emb, "threshold": threshold})

        if reply:
            log.info("🎯 find_related_goals():", f"{reply.total} matches, {len(reply.docs)} <= {threshold:0.3f}")
            return reply.docs

        log.info("🎯 find_related_goals(): no match")
        return []
