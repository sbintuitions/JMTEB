from __future__ import annotations

import json
from abc import ABC, abstractmethod

import datasets
import smart_open
from loguru import logger
from pydantic.dataclasses import dataclass


@dataclass
class RerankingQuery:
    query: str
    retrieved_docs: list[str | int]
    relevance_scores: list[int]


@dataclass
class RerankingDoc:
    id: str | int
    text: str


@dataclass
class RerankingPrediction:
    query: str
    relevant_docs: list[RerankingDoc]
    reranked_relevant_docs: list[RerankingDoc]


class RerankingQueryDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RerankingQuery:
        pass

    def __eq__(self, __value: object) -> bool:
        return False


class RerankingDocDataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx) -> RerankingDoc:
        pass

    def __eq__(self, __value: object) -> bool:
        return False

    def _build_idx_docid_mapping(self, dataset_attr_name: str = "dataset") -> None:
        self.idx_to_docid: dict = {}
        self.docid_to_idx: dict = {}
        id_key: str = getattr(self, "id_key", None)
        dataset = getattr(self, dataset_attr_name)
        if id_key:
            for idx, doc_dict in enumerate(dataset):
                self.idx_to_docid[idx] = doc_dict[id_key]
                self.docid_to_idx[doc_dict[id_key]] = idx
        elif isinstance(dataset[0], RerankingDoc):
            for idx, doc in enumerate(dataset):
                doc: RerankingDoc
                self.idx_to_docid[idx] = doc.id
                self.docid_to_idx[doc.id] = idx
        else:
            raise ValueError(f"Invalid dataset type: list[{type(dataset[0])}]")


class HfRerankingQueryDataset(RerankingQueryDataset):
    def __init__(
        self,
        path: str,
        split: str,
        name: str | None = None,
        query_key: str = "query",
        retrieved_docs_key: str = "retrieved_docs",
        relevance_scores_key: str = "relevance_scores",
    ):
        self.path = path
        self.split = split
        self.name = name
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.query_key = query_key
        self.retrieved_docs_key = retrieved_docs_key
        self.relevance_scores_key = relevance_scores_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingQuery:
        retrieved_docs = self.dataset[idx][self.retrieved_docs_key]
        relevance_scores = self.dataset[idx][self.relevance_scores_key]

        return RerankingQuery(
            query=self.dataset[idx][self.query_key], retrieved_docs=retrieved_docs, relevance_scores=relevance_scores
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "query_key", "retrieved_docs_key", "relevance_scores_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRerankingQueryDataset(RerankingQueryDataset):
    def __init__(
        self,
        filename: str,
        query_key: str = "query",
        retrieved_docs_key: str = "retrieved_docs",
        relevance_scores_key: str = "relevance_scores",
    ):
        self.filename = filename
        self.dataset: datasets.Dataset = datasets.load_dataset("json", data_files=filename)["train"]
        self.query_key = query_key
        self.retrieved_docs_key = retrieved_docs_key
        self.relevance_scores_key = relevance_scores_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingQuery:
        retrieved_docs = self.dataset[idx][self.retrieved_docs_key]
        relevance_scores = self.dataset[idx][self.relevance_scores_key]

        return RerankingQuery(
            query=self.dataset[idx][self.query_key], retrieved_docs=retrieved_docs, relevance_scores=relevance_scores
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "query_key", "retrieved_docs_key", "relevance_scores_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class HfRerankingDocDataset(RerankingDocDataset):
    def __init__(self, path: str, split: str, name: str | None = None, id_key: str = "docid", text_key: str = "text"):
        logger.info(f"Loading dataset {path} (name={name}) with split {split}")
        self.path = path
        self.split = split
        self.name = name
        self.dataset = datasets.load_dataset(path, split=split, name=name, trust_remote_code=True)
        self.id_key = id_key
        self.text_key = text_key
        self._build_idx_docid_mapping()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingDoc:
        return RerankingDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("path", "split", "name", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True


class JsonlRerankingDocDataset(RerankingDocDataset):
    def __init__(self, filename: str, id_key: str = "docid", text_key: str = "text"):
        logger.info(f"Loading dataset from {filename}")
        self.filename = filename
        with smart_open.open(filename, "r", encoding="utf-8", errors="ignore") as fin:
            corpus = [json.loads(line.strip()) for line in fin.readlines()]
        self.dataset = corpus
        self.id_key = id_key
        self.text_key = text_key
        self._build_idx_docid_mapping()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> RerankingDoc:
        return RerankingDoc(id=self.dataset[idx][self.id_key], text=self.dataset[idx][self.text_key].strip())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        for attribute in ("filename", "id_key", "text_key"):
            if getattr(self, attribute, None) != getattr(other, attribute, None):
                return False
        return True
