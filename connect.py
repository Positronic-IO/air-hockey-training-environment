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
                    if isinstance(value, list) and any(
                        [item.isdigit() for item in value]
                    ):
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
    def get(self):
        pass

    @abstractmethod
    def post(self):
        pass


class RedisConnection(BaseConnection):

    redis = StrictRedis()

    def __init__(self):
        pass

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
            print(obj)
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

    def post(self, payload: Dict[str, Union[str, int, List[Any]]]):
        """ Save data to Redis """

        for key, value in payload.items():
            self.redis.set(key, self.serialize(value))


class ServerConnection(BaseConnection):
    def __init__(self, test: bool = False):

        url = "http://localhost:8000/api/environment"
        test_url = "http://localhost:8000/api/test/environment"

        if not test:
            self.endpoint = url
        else:
            self.endpoint = test_url

    def get(self, key: Union[str, List[str]] = ""):
        """ Grab object(s) from server """

        # If key is not passed, retrieve everything from serevr
        if not key:
            r = requests.get(self.endpoint)
            if r.status_code >= 400:
                return ""
            return r.json()

        # If we search based on a single key
        if key and isinstance(key, str):
            r = requests.get(self.endpoint, params={"key": key})
            if r.status_code >= 400:
                print(f"Object {key} not found")
                return ""
            return r.json()

        # If we search based on a multiple keys
        if key and isinstance(key, list):
            data = dict()
            for item in key:
                r = requests.get(self.endpoint, params={"key": item})
                if r.status_code >= 400:
                    print(f"Object {item} not found")
                    continue
                data = {**data, **r.json()}
            return data

        return ""

    def post(self, data: Dict[str, Union[str, int, List[Any]]]):
        """ Save data to Server """
        r = requests.post(self.endpoint, data=data)
        if r.status_code >= 400:
            print(f"Error")
            return False
        return True


class Connection:

    """ Connection facade
    
        We could connect to either redis or local server. 
    """

    connections = {"redis": RedisConnection, "server": ServerConnection}

    def __init__(self):
        pass

    def make(self, connection: str, test: bool = False):

        if connection == "redis":
            return self.connections["redis"]()

        if connection == "server":
            return self.connections["server"](test)

        raise TypeError("Connection not found")
