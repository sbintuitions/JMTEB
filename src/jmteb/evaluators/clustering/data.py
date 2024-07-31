from __future__ import annotations

from abc import ABC, abstractmethod

import datasets
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass
class ClusteringInstance:
    text: str
    label: int


@dataclass
class ClusteringPrediction:
    text: str
    label: int
    prediction: int


class ClusteringDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> ClusteringInstance:
        pass

    def __eq__(self, __value: object) -> bool:
        return False


class HfClusteringDataset(ClusteringDataset):
    def __init__(
        self, path: str, split: str, name: str | None = None, text_key: str = "text", label_key: str = "label"
    ):
        logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.path = path
        self.split = split
        self.name = name
        self.dataset: datasets.Dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.text_key = text_key
        self.label_key = label_key
        if not self.dataset.features[self.label_key].dtype.startswith("int"):
            label_to_int = {label: i for i, label in enumerate(sorted(set(self.dataset[self.label_key])))}
            self.dataset = self.dataset.map(lambda example: {"label": label_to_int[example[label_key]]})
            self.label_to_int = label_to_int

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> ClusteringInstance:
        return ClusteringInstance(text=self.dataset[idx][self.text_key], label=self.dataset[idx][self.label_key])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "text_key", "label_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlClusteringDataset(ClusteringDataset):
    def __init__(self, filename: str, text_key: str = "text", label_key: str = "label") -> None:
        logger.info(f"Loading dataset from {filename}")
        self.filename = filename
        self.dataset: datasets.Dataset = datasets.load_dataset("json", data_files=filename)["train"]
        self.text_key = text_key
        self.label_key = label_key
        if not self.dataset.features[self.label_key].dtype.startswith("int"):
            label_to_int = {label: i for i, label in enumerate(sorted(set(self.dataset[self.label_key])))}
            self.dataset = self.dataset.map(lambda example: {"label": label_to_int[example[label_key]]})
            self.label_to_int = label_to_int

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> ClusteringInstance:
        return ClusteringInstance(text=self.dataset[idx][self.text_key], label=self.dataset[idx][self.label_key])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "text_key", "label_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True
