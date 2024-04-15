from __future__ import annotations

import json
from abc import ABC, abstractmethod

import datasets
import smart_open
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass
class RetrievalQuery:
    query: str
    relevant_docs: list[str | int]


@dataclass
class RetrievalDoc:
    id: str | int
    text: str


class RetrievalQueryDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RetrievalQuery:
        pass

    def __eq__(self, __value: object) -> bool:
        return False


class RetrievalDocDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RetrievalDoc:
        pass

    def __eq__(self, __value: object) -> bool:
        return False


class HfRetrievalQueryDataset(RetrievalQueryDataset):
    def __init__(
        self,
        path: str,
        split: str,
        name: str | None = None,
        query_key: str = "query",
        relevant_docs_key: str = "relevant_docs",
    ):
        self.path = path
        self.split = split
        self.name = name
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.query_key = query_key
        self.relevant_docs_key = relevant_docs_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalQuery:
        relevant_docs = self.dataset[idx][self.relevant_docs_key]
        if not isinstance(relevant_docs, list):
            relevant_docs = [relevant_docs]

        return RetrievalQuery(query=self.dataset[idx][self.query_key], relevant_docs=relevant_docs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "query_key", "retrieved_docs_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRetrievalQueryDataset(RetrievalQueryDataset):
    def __init__(
        self,
        filename: str,
        query_key: str = "query",
        relevant_docs_key: str = "relevant_docs",
    ):
        self.filename = filename
        self.dataset: datasets.Dataset = datasets.load_dataset("json", data_files=filename)["train"]
        self.query_key = query_key
        self.relevant_docs_key = relevant_docs_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalQuery:
        relevant_docs = self.dataset[idx][self.relevant_docs_key]
        if not isinstance(relevant_docs, list):
            relevant_docs = [relevant_docs]

        return RetrievalQuery(query=self.dataset[idx][self.query_key], relevant_docs=relevant_docs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "query_key", "relevance_scores_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class HfRetrievalDocDataset(RetrievalDocDataset):
    def __init__(self, path: str, split: str, name: str | None = None, id_key: str = "docid", text_key: str = "text"):
        logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.path = path
        self.split = split
        self.name = name
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.id_key = id_key
        self.text_key = text_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalDoc:
        return RetrievalDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRetrievalDocDataset(RetrievalDocDataset):
    def __init__(self, filename: str, id_key: str = "docid", text_key: str = "text"):
        logger.info(f"Loading dataset from {filename}")
        self.filename = filename
        with smart_open.open(filename, "r", encoding="utf-8", errors="ignore") as fin:
            corpus = [json.loads(line.strip()) for line in fin.readlines()]
        self.dataset = corpus
        self.id_key = id_key
        self.text_key = text_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RetrievalDoc:
        return RetrievalDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key].strip())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True
