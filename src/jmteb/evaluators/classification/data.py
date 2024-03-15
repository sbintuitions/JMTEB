from __future__ import annotations

from abc import ABC, abstractmethod

import datasets
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass
class ClassificationInstance:
    text: str
    label: int


class ClassificationDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> ClassificationInstance:
        pass


class HfClassificationDataset(ClassificationDataset):
    def __init__(
        self, path: str, split: str, name: str | None = None, text_key: str = "text", label_key: str = "label"
    ):
        logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.text_key = text_key
        self.label_key = label_key
        if not self.dataset.features[self.label_key].dtype.startswith("int"):
            label_to_int = {label: i for i, label in enumerate(sorted(set(self.dataset[self.label_key])))}
            self.dataset = self.dataset.map(lambda example: {"label": label_to_int[example[label_key]]})
            self.label_to_int = label_to_int

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> ClassificationInstance:
        return ClassificationInstance(text=self.dataset[idx][self.text_key], label=self.dataset[idx][self.label_key])


class JsonlClassificationDataset(ClassificationDataset):
    def __init__(self, filename: str, text_key: str = "text", label_key: str = "label") -> None:
        logger.info(f"Loading dataset from {filename}")
        self.dataset: datasets.Dataset = datasets.load_dataset("json", data_files=filename)["train"]
        self.text_key = text_key
        self.label_key = label_key
        if not self.dataset.features[self.label_key].dtype.startswith("int"):
            label_to_int = {label: i for i, label in enumerate(sorted(set(self.dataset[self.label_key])))}
            self.dataset = self.dataset.map(lambda example: {"label": label_to_int[example[label_key]]})
            self.label_to_int = label_to_int

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> ClassificationInstance:
        return ClassificationInstance(text=self.dataset[idx][self.text_key], label=self.dataset[idx][self.label_key])
