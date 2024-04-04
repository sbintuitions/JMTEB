from __future__ import annotations

import warnings
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch
import tqdm
from loguru import logger
from torch import Tensor

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import RerankingDocDataset, RerankingQueryDataset

T = TypeVar("T")


class RerankingEvaluator(EmbeddingEvaluator):
    """
    Evaluator for reranking task.

    Args:
        query_dataset (RerankingQueryDataset): query dataset
        doc_dataset (RerankingDocDataset): document dataset
        ndcg_at_k (list[int] | None): top k documents to consider in NDCG (Normalized Documented Cumulative Gain).
    """

    def __init__(
        self,
        query_dataset: RerankingQueryDataset,
        doc_dataset: RerankingDocDataset,
        ndcg_at_k: list[int] | None = None,
    ) -> None:
        self.query_dataset = query_dataset
        self.doc_dataset = doc_dataset
        self.ndcg_at_k = ndcg_at_k or [10, 20]
        self.main_metric = f"ndcg@{self.ndcg_at_k[0]}"

    def __call__(
        self,
        model: TextEmbedder,
        cache_dir: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
    ) -> EvaluationResults:
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        query_embeddings = model.batch_encode_with_cache(
            text_list=[item.query for item in self.query_dataset],
            cache_path=Path(cache_dir) / "query.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )

        doc_embeddings = model.batch_encode_with_cache(
            text_list=[item.text for item in self.doc_dataset],
            cache_path=Path(cache_dir) / "corpus.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )

        doc_indices = {item.id: i for i, item in enumerate(self.doc_dataset)}

        logger.info("Start reranking")
        results: dict[str, dict[str, float]] = {}

        dist_metrics: dict[str, Callable] = {
            "cosine_similarity": Similarities.cosine_similarity,
            "dot_score": Similarities.dot_score,
            "euclidean_distance": Similarities.euclidean_distance,
        }

        for dist_metric, dist_func in dist_metrics.items():
            dist_scores: dict[str, float] = {}

            with tqdm.tqdm(total=len(self.query_dataset), desc="Reranking docs") as pbar:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                reranked_docs_list = []
                for i, item in enumerate(self.query_dataset):
                    query_embedding = convert_to_tensor(query_embeddings[i], device=device)
                    doc_embedding = convert_to_tensor(np.array([doc_embeddings[doc_indices[retrieved_doc]] for retrieved_doc in item.retrieved_docs]), device=device)
                    similarity = dist_func(query_embedding, doc_embedding)

                    argsorted_indices = torch.argsort(
                        similarity,
                        dim=1,
                        descending=True,
                    )[0]
                    reranked_docs = [item.retrieved_docs[argsorted_indice] for argsorted_indice in argsorted_indices]
                    reranked_docs_list.append(reranked_docs)
                    pbar.update(i)
            
            retrieved_docs_list = [item.retrieved_docs for item in self.query_dataset]
            relevance_scores_list = [item.relevance_scores for item in self.query_dataset]

            for k in self.ndcg_at_k:
                dist_scores[f"ndcg@{k}"] = ndcg_at_k(retrieved_docs_list, relevance_scores_list, reranked_docs_list, k)

            results[dist_metric] = dist_scores

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=max([v[self.main_metric] for v in results.values()]),
            details=results,
        )


def ndcg_at_k(retrieved_docs_list: list[list[T]], relevance_scores_list: list[list[T]], reranked_docs_list: list[list[T]], k: int) -> float:
    total_ndcg_scores = 0
    for retrieved_docs, relevance_scores, reranked_docs in zip(retrieved_docs_list, relevance_scores_list, reranked_docs_list):
        dcg = 0
        for rank, doc_id in enumerate(reranked_docs[:k], start=1):
            relevance_score = relevance_scores[retrieved_docs.index(doc_id)]
            dcg += relevance_score / np.log2(rank + 1)
        idcg = sum([relevance_score / np.log2(rank + 1) for rank, relevance_score in enumerate(sorted(relevance_scores)[::-1][:k], start=1)])
        total_ndcg_scores += dcg / idcg
    return total_ndcg_scores / len(retrieved_docs_list)


def convert_to_tensor(embeddings: np.ndarray | Tensor, device: str) -> Tensor:
    if not isinstance(embeddings, Tensor):
        embeddings = torch.tensor(embeddings)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.unsqueeze(0)
    return embeddings.to(device=device)


@dataclass
class Similarities:
    @staticmethod
    def cosine_similarity(e1: Tensor, e2: Tensor) -> Tensor:
        e1_norm = torch.nn.functional.normalize(e1, p=2, dim=1)
        e2_norm = torch.nn.functional.normalize(e2, p=2, dim=1)
        return torch.mm(e1_norm, e2_norm.transpose(0, 1))

    @staticmethod
    def manhatten_distance(e1: Tensor, e2: Tensor) -> Tensor:
        # the more distant, the less similar, so we use 100 / dist as similarity
        x = e1.unsqueeze(1)
        y = e2.unsqueeze(0).repeat(e1.shape[0], 1, 1)
        return 100 / ((x - y).abs().sum(dim=-1) + 1e-4)

    @staticmethod
    def euclidean_distance(e1: Tensor, e2: Tensor) -> Tensor:
        # the more distant, the less similar, so we use 100 / dist as similarity
        return 100 / (torch.cdist(e1, e2) + 1e-4)

    @staticmethod
    def dot_score(e1: Tensor, e2: Tensor) -> Tensor:
        return torch.mm(e1, e2.transpose(0, 1))
