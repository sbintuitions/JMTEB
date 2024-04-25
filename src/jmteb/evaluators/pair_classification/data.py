from __future__ import annotations

from abc import ABC, abstractmethod

import datasets
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass
class PairClassificationInstance:
    sentence1: str
    sentence2: str
    label: int


class PairClassificationDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> PairClassificationInstance:
        pass

    def __eq__(self, __value: object) -> bool:
        return False


class HfPairClassificationDataset(PairClassificationDataset):
    def __init__(
        self,
        path: str,
        split: str,
        name: str | None = None,
        sentence1_key: str = "sentence1",
        sentence2_key: str = "sentence2",
        label_key: str = "label",
    ):
        logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.path = path
        self.split = split
        self.name = name
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.label_key = label_key
        if not self.dataset.features[self.label_key].dtype.startswith("int"):
            label_to_int = {label: i for i, label in enumerate(sorted(set(self.dataset[self.label_key])))}
            self.dataset = self.dataset.map(lambda example: {"label": label_to_int[example[label_key]]})
            self.label_to_int = label_to_int

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> PairClassificationInstance:
        return PairClassificationInstance(
            sentence1=self.dataset[idx][self.sentence1_key],
            sentence2=self.dataset[idx][self.sentence2_key],
            label=self.dataset[idx][self.label_key],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "sentence1_key", "sentence2_key", "label_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlPairClassificationDataset(PairClassificationDataset):
    def __init__(
        self,
        filename: str,
        sentence1_key: str = "sentence1",
        sentence2_key: str = "sentence2",
        label_key: str = "label",
    ) -> None:
        logger.info(f"Loading dataset from {filename}")
        self.filename = filename
        self.dataset: datasets.Dataset = datasets.load_dataset("json", data_files=filename)["train"]
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.label_key = label_key
        if not self.dataset.features[self.label_key].dtype.startswith("int"):
            label_to_int = {label: i for i, label in enumerate(sorted(set(self.dataset[self.label_key])))}
            self.dataset = self.dataset.map(lambda example: {"label": label_to_int[example[label_key]]})
            self.label_to_int = label_to_int

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> PairClassificationInstance:
        return PairClassificationInstance(
            sentence1=self.dataset[idx][self.sentence1_key],
            sentence2=self.dataset[idx][self.sentence2_key],
            label=self.dataset[idx][self.label_key],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "sentence1_key", "sentence2_key", "label_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True
