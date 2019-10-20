""" Connection Facade """

import json
from typing import Any, Dict, List, Union

from redis import ConnectionError, StrictRedis


class RedisConnection:
    def __init__(self):
        self.redis = StrictRedis()
        self.p = self.redis.pubsub(ignore_subscribe_messages=True)

    def get(self, key: str = ""):
        """ Grab object from Redis and serialize """
        try:
            obj = self.redis.get(key)
        except ConnectionError:
            print(f"Object {key} not in Redis")
            return ""
        return {key: json.loads(obj)}

    def post(self, payload: Any):
        """ Save data to Redis """

        for key, value in payload.items():
            self.redis.set(key, json.dumps(value))

    def publish(self, channel: str, payload: Any = True):
        """ Publish message """
        self.redis.publish(channel, json.dumps(payload))