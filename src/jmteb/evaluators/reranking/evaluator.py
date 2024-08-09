from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch
import tqdm
from loguru import logger
from torch import Tensor
from torch import distributed as dist

from jmteb.embedders.base import TextEmbedder
from jmteb.evaluators.base import EmbeddingEvaluator, EvaluationResults

from .data import (
    RerankingDoc,
    RerankingDocDataset,
    RerankingPrediction,
    RerankingQuery,
    RerankingQueryDataset,
)

T = TypeVar("T")


class RerankingEvaluator(EmbeddingEvaluator):
    """
    Evaluator for reranking task.

    Args:
        val_query_dataset (RerankingQueryDataset): validation query dataset used for hyperparameter tuning
        test_query_dataset (RerankingQueryDataset): test query dataset used for computing the scores
        doc_dataset (RerankingDocDataset): document dataset
        ndcg_at_k (list[int] | None): top k documents to consider in NDCG (Normalized Documented Cumulative Gain).
        query_prefix (str | None): prefix for queries. Defaults to None.
        doc_prefix (str | None): prefix for documents. Defaults to None.
        log_predictions (bool): whether to log predictions of each datapoint. Defaults to False.
        top_n_docs_to_log (int): log only top n documents. Defaults to 5.
    """

    def __init__(
        self,
        val_query_dataset: RerankingQueryDataset,
        test_query_dataset: RerankingQueryDataset,
        doc_dataset: RerankingDocDataset,
        ndcg_at_k: list[int] | None = None,
        query_prefix: str | None = None,
        doc_prefix: str | None = None,
        log_predictions: bool = False,
        top_n_docs_to_log: int = 5,
    ) -> None:
        self.test_query_dataset = test_query_dataset
        self.val_query_dataset = val_query_dataset
        self.doc_dataset = doc_dataset
        self.ndcg_at_k = ndcg_at_k or [10, 20, 40]
        self.main_metric = f"ndcg@{self.ndcg_at_k[0]}"
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.log_predictions = log_predictions
        self.top_n_docs_to_log = top_n_docs_to_log

    def __call__(
        self,
        model: TextEmbedder,
        cache_dir: str | PathLike[str] | None = None,
        overwrite_cache: bool = False,
    ) -> EvaluationResults:
        model.set_output_tensor()
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        val_query_embeddings = model.batch_encode_with_cache(
            text_list=[item.query for item in self.val_query_dataset],
            prefix=self.query_prefix,
            cache_path=Path(cache_dir) / "val_query.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )
        if self.val_query_dataset == self.test_query_dataset:
            test_query_embeddings = val_query_embeddings
        else:
            test_query_embeddings = model.batch_encode_with_cache(
                text_list=[item.query for item in self.test_query_dataset],
                prefix=self.query_prefix,
                cache_path=Path(cache_dir) / "test_query.bin" if cache_dir is not None else None,
                overwrite_cache=overwrite_cache,
            )
        doc_embeddings = model.batch_encode_with_cache(
            text_list=[item.text for item in self.doc_dataset],
            prefix=self.doc_prefix,
            cache_path=Path(cache_dir) / "corpus.bin" if cache_dir is not None else None,
            overwrite_cache=overwrite_cache,
        )

        logger.info("Start reranking")

        dist_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
            "cosine_similarity": Similarities.cosine_similarity,
            "dot_score": Similarities.dot_score,
            "euclidean_distance": Similarities.euclidean_distance,
        }

        val_results = {}
        for dist_name, dist_func in dist_functions.items():
            val_results[dist_name], _ = self._compute_metrics(
                query_dataset=self.val_query_dataset,
                query_embeddings=val_query_embeddings,
                doc_embeddings=doc_embeddings,
                dist_func=dist_func,
            )

        sorted_val_results = sorted(val_results.items(), key=lambda res: res[1][self.main_metric], reverse=True)
        optimal_dist_name = sorted_val_results[0][0]
        scores, reranked_docs_list = self._compute_metrics(
            query_dataset=self.test_query_dataset,
            query_embeddings=test_query_embeddings,
            doc_embeddings=doc_embeddings,
            dist_func=dist_functions[optimal_dist_name],
        )
        test_results = {optimal_dist_name: scores}

        return EvaluationResults(
            metric_name=self.main_metric,
            metric_value=test_results[optimal_dist_name][self.main_metric],
            details={
                "optimal_distance_metric": optimal_dist_name,
                "val_scores": val_results,
                "test_scores": test_results,
            },
            predictions=(
                self._format_predictions(
                    self.test_query_dataset, self.doc_dataset, reranked_docs_list, self.top_n_docs_to_log
                )
                if self.log_predictions
                else None
            ),
        )

    def _compute_metrics(
        self,
        query_dataset: RerankingQueryDataset,
        query_embeddings: np.ndarray | Tensor,
        doc_embeddings: np.ndarray | Tensor,
        dist_func: Callable[[Tensor, Tensor], Tensor],
    ) -> tuple[dict[str, float], list[list[str | int]]]:
        doc_indices = {item.id: i for i, item in enumerate(self.doc_dataset)}

        results: dict[str, float] = {}

        with tqdm.tqdm(total=len(query_dataset), desc="Reranking docs") as pbar:
            if torch.cuda.is_available():
                if dist.is_torchelastic_launched():
                    device = f"cuda:{dist.get_rank()}"
                else:
                    device = "cuda"
            else:
                device = "cpu"
            reranked_docs_list = []
            for i, item in enumerate(query_dataset):
                query_embedding = to_tensor(query_embeddings[i], device=device)
                doc_embedding = torch.stack(
                    [
                        Tensor(doc_embeddings[doc_indices[retrieved_doc]]).to(device=device)
                        for retrieved_doc in item.retrieved_docs
                    ]
                )
                similarity = dist_func(query_embedding, doc_embedding)

                argsorted_indices = torch.argsort(
                    similarity,
                    dim=1,
                    descending=True,
                )[0]
                reranked_docs = [item.retrieved_docs[argsorted_indice] for argsorted_indice in argsorted_indices]
                reranked_docs_list.append(reranked_docs)
                pbar.update(i)

        retrieved_docs_list = [item.retrieved_docs for item in query_dataset]
        relevance_scores_list = [item.relevance_scores for item in query_dataset]

        for k in self.ndcg_at_k:
            results[f"ndcg@{k}"] = ndcg_at_k(retrieved_docs_list, relevance_scores_list, reranked_docs_list, k)

        return results, reranked_docs_list

    @staticmethod
    def _format_predictions(
        query_dataset: RerankingQueryDataset,
        doc_dataset: RerankingDocDataset,
        reranked_docs_list: list[list],
        top_n_to_log: int,
    ) -> list[RerankingPrediction]:
        predictions = []
        for q, pred_docids in zip(query_dataset, reranked_docs_list):
            q: RerankingQuery
            golden_docs: list[RerankingDoc] = [
                doc_dataset[doc_dataset.docid_to_idx[docid]] for docid in q.retrieved_docs
            ]
            pred_docids = pred_docids[:top_n_to_log]
            pred_docs: list[RerankingDoc] = [
                doc_dataset[doc_dataset.docid_to_idx[pred_docid]] for pred_docid in pred_docids
            ]
            logger.info(f"{golden_docs=}")
            logger.info(f"{pred_docs=}")
            prediction = RerankingPrediction(
                query=q.query,
                relevant_docs=golden_docs,
                reranked_relevant_docs=pred_docs,
            )
            predictions.append(prediction)
        return predictions


def ndcg_at_k(
    retrieved_docs_list: list[list[T]], relevance_scores_list: list[list[T]], reranked_docs_list: list[list[T]], k: int
) -> float:
    total_ndcg_scores = 0
    for retrieved_docs, relevance_scores, reranked_docs in zip(
        retrieved_docs_list, relevance_scores_list, reranked_docs_list
    ):
        dcg = 0
        for rank, doc_id in enumerate(reranked_docs[:k], start=1):
            relevance_score = relevance_scores[retrieved_docs.index(doc_id)]
            dcg += relevance_score / np.log2(rank + 1)
        idcg = sum(
            [
                relevance_score / np.log2(rank + 1)
                for rank, relevance_score in enumerate(sorted(relevance_scores)[::-1][:k], start=1)
            ]
        )
        total_ndcg_scores += dcg / idcg
    return total_ndcg_scores / len(retrieved_docs_list)


def to_tensor(embeddings: np.ndarray | Tensor, device: str) -> Tensor:
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
