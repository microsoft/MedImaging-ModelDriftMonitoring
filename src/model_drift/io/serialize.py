#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
from abc import ABCMeta

import numpy as np
import yaml


class SerializableMeta(ABCMeta):
    REGISTRY = {}

    def __new__(cls, name, bases, namespace, **kwargs):
        # instantiate a new type corresponding to the type of class being defined
        # this is currently RegisterBase but in child classes will be the child class
        new_cls = super().__new__(cls, name, bases, namespace, **kwargs)
        cls.REGISTRY[new_cls.__name__.lower()] = new_cls
        new_cls.yaml_tag = f"!{new_cls.__name__.lower()}"
        return new_cls

    @classmethod
    def get_registry(cls):
        return dict(cls.REGISTRY)


def get_dumper():
    safe_dumper = yaml.SafeDumper
    for cls in SerializableMeta.get_registry().values():
        representer = lambda dumper, instance: dumper.represent_mapping(instance.yaml_tag, instance.serialize())
        safe_dumper.add_representer(cls, representer)
    return safe_dumper


def get_loader():
    loader = yaml.SafeLoader

    for cls in SerializableMeta.get_registry().values():
        constructor = lambda loader, node: cls.deserialize(loader.construct_mapping(node))
        loader.add_constructor(cls.yaml_tag, constructor)
    return loader


class SerializableBase(metaclass=SerializableMeta):

    def serialize(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def deserialize(cls, dct):
        obj = cls()
        for k, v in dct.items():
            obj.__dict__[k] = v
        return obj


class ModelDriftEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, SerializableBase):
            return {"__cls__": type(self).__name__.lower(), **obj.serialize()}
        return super().default(obj)


class ModelDriftDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "__cls__" in dct:
            cls = SerializableBase.get_registry()[dct.pop('__cls__')]
            return cls.deserialize(dct)
        return dct
