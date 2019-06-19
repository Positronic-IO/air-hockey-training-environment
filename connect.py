""" Connection Facade """

import json
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Any, Dict, List, Union

import requests
from redis import ConnectionError, StrictRedis


class BaseConnection(ABC):
    def __init__(self):
        pass

    @staticmethod
    def serialize(payload):
        return json.dumps(payload)

    @staticmethod
    def deserialize(payload):
        data = json.loads(payload)

        # Assert that coordinates are ints
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    if isinstance(value, list) and any([item.isdigit() for item in value]):
                        data[key] = list(map(int, value))
                except AttributeError:
                    continue

        try:
            if isinstance(data, list) and any([item.isdigit() for item in data]):
                data = list(map(int, data))
        except AttributeError:
            return data

        return data

    @abstractmethod
    def get(self, key: Union[str, List[str]]):
        pass

    @abstractmethod
    def post(self, payload: Any):
        pass


class RedisConnection(BaseConnection):
    def __init__(self):
        self.redis = StrictRedis()
        self.p = self.redis.pubsub(ignore_subscribe_messages=True)

    def get(self, key: Union[str, List[str]] = ""):
        """ Grab object(s) from Redis and serialize """

        # If there is not a key, show everyting
        if not key:
            data = dict()
            if self.redis.exists("puck"):
                data["puck"] = self.deserialize(self.redis.get("puck"))

            # if self.redis.exists("lidar"):
            #     data["lidar"] = self.deserialize(self.redis.get("lidar"))

            if self.redis.exists("robot"):
                data["robot"] = self.deserialize(self.redis.get("robot"))

            if self.redis.exists("opponent"):
                data["opponent"] = self.deserialize(self.redis.get("opponent"))

            return data

        # Search by specific key
        if isinstance(key, str):
            try:
                obj = self.redis.get(key)
            except ConnectionError:
                print(f"Object {key} not in Redis")
                return ""
            return {key: self.deserialize(obj)}

        # Search for items in a list
        if isinstance(key, list):
            data = dict()
            for item in key:
                try:
                    obj = self.redis.get(item)
                except ConnectionError:
                    print(f"Object {item} not in Redis")
                    continue
                data[item] = self.deserialize(obj)
            return data

        return ""

    def post(self, payload: Any):
        """ Save data to Redis """

        for key, value in payload.items():
            self.redis.set(key, self.serialize(value))

    def publish(self, channel):
        self.redis.publish("update", self.serialize(True))
