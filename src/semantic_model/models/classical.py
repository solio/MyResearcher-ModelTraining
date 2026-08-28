from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from ..errors import ContractError
from ..schema import SINGLE_LABEL_HEADS, V1_HEADS, LabelSchema


def _vectorizer_args(raw: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    if "ngram_range" in result:
        result["ngram_range"] = tuple(result["ngram_range"])
    if result.get("dtype") == "float32":
        result["dtype"] = np.float32
    return result


def build_vectorizer(model_config: Mapping[str, Any]) -> FeatureUnion:
    try:
        vectorizers = {
            "word_tfidf": TfidfVectorizer(
                **_vectorizer_args(model_config["word_tfidf"])
            ),
            "char_tfidf": TfidfVectorizer(
                **_vectorizer_args(model_config["char_tfidf"])
            ),
        }
        order = model_config.get(
            "feature_stack_order", ["word_tfidf", "char_tfidf"]
        )
        if not isinstance(order, Sequence) or set(order) != set(vectorizers):
            raise ContractError(
                "MODEL_CONFIG_INVALID",
                "feature_stack_order must contain char_tfidf and word_tfidf once",
            )
    except (KeyError, TypeError) as exc:
        raise ContractError(
            "MODEL_CONFIG_INVALID", "invalid TF-IDF configuration"
        ) from exc
    return FeatureUnion(
        tuple((str(name).removesuffix("_tfidf"), vectorizers[str(name)]) for name in order)
    )


def _build_logistic(config: Mapping[str, Any]) -> LogisticRegression:
    allowed = {
        "C",
        "max_iter",
        "solver",
        "class_weight",
        "random_state",
        "tol",
        "penalty",
    }
    return LogisticRegression(**{key: value for key, value in config.items() if key in allowed})


@dataclass
class SingleLabelEstimator:
    class_order: list[str]
    estimator: LogisticRegression | None = None
    constant_label: str | None = None

    def fit(
        self,
        matrix: sparse.spmatrix,
        labels: Sequence[str],
        weights: Sequence[float],
        logistic_config: Mapping[str, Any],
    ) -> None:
        observed = sorted(set(labels))
        if not set(observed) <= set(self.class_order):
            raise ContractError(
                "UNKNOWN_LABEL_VALUE", "training label is outside frozen class order"
            )
        if len(observed) == 1:
            self.constant_label = observed[0]
            return
        self.estimator = _build_logistic(logistic_config)
        self.estimator.fit(matrix, labels, sample_weight=np.asarray(weights, dtype=float))

    def predict_proba(self, matrix: sparse.spmatrix) -> np.ndarray:
        result = np.zeros((matrix.shape[0], len(self.class_order)), dtype=float)
        if self.constant_label is not None:
            result[:, self.class_order.index(self.constant_label)] = 1.0
            return result
        if self.estimator is None:
            raise ContractError("MODEL_NOT_FITTED", "single-label estimator is not fitted")
        native = self.estimator.predict_proba(matrix)
        for native_index, label in enumerate(self.estimator.classes_):
            result[:, self.class_order.index(str(label))] = native[:, native_index]
        return result


@dataclass
class BinaryTagEstimator:
    estimator: LogisticRegression | None = None
    constant: int | None = None

    def fit(
        self,
        matrix: sparse.spmatrix,
        labels: Sequence[int],
        weights: Sequence[float],
        logistic_config: Mapping[str, Any],
    ) -> None:
        observed = sorted(set(int(value) for value in labels))
        if len(observed) == 1:
            self.constant = observed[0]
            return
        self.estimator = _build_logistic(logistic_config)
        self.estimator.fit(
            matrix,
            np.asarray(labels, dtype=int),
            sample_weight=np.asarray(weights, dtype=float),
        )

    def predict_positive_probability(self, matrix: sparse.spmatrix) -> np.ndarray:
        if self.constant is not None:
            return np.full(matrix.shape[0], float(self.constant), dtype=float)
        if self.estimator is None:
            raise ContractError("MODEL_NOT_FITTED", "binary estimator is not fitted")
        native = self.estimator.predict_proba(matrix)
        positive_indices = np.where(self.estimator.classes_ == 1)[0]
        if len(positive_indices) != 1:
            raise ContractError(
                "MODEL_CLASS_ORDER_INVALID", "binary estimator has no positive class"
            )
        return native[:, int(positive_indices[0])]


@dataclass
class ClassicalMultiHeadModel:
    schema_version: str
    class_order: dict[str, list[str]]
    model_config: Mapping[str, Any]
    vectorizer: FeatureUnion
    single_heads: dict[str, SingleLabelEstimator]
    reasoning_heads: dict[str, BinaryTagEstimator]
    fitted: bool = False

    @classmethod
    def create(
        cls, schema: LabelSchema, model_config: Mapping[str, Any]
    ) -> "ClassicalMultiHeadModel":
        if model_config.get("family") != "tfidf-logistic-regression":
            raise ContractError(
                "MODEL_CONFIG_INVALID", "only the transparent classical family is allowed"
            )
        return cls(
            schema_version=schema.schema_version,
            class_order={head: list(order) for head, order in schema.class_order.items()},
            model_config=dict(model_config),
            vectorizer=build_vectorizer(model_config),
            single_heads={
                head: SingleLabelEstimator(list(schema.class_order[head]))
                for head in SINGLE_LABEL_HEADS
            },
            reasoning_heads={
                tag: BinaryTagEstimator()
                for tag in schema.class_order["reasoning_tags"]
            },
        )

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[Mapping[str, Any]],
        field_weights: Sequence[Mapping[str, float]],
    ) -> "ClassicalMultiHeadModel":
        if not texts or not (len(texts) == len(labels) == len(field_weights)):
            raise ContractError(
                "TRAINING_INPUT_INVALID", "texts, labels, and weights must align"
            )
        matrix = self.vectorizer.fit_transform(texts)
        scalar_logistic_config = self.model_config.get(
            "scalar_logistic_regression",
            self.model_config.get("logistic_regression", {}),
        )
        reasoning_logistic_config = self.model_config.get(
            "reasoning_logistic_regression",
            self.model_config.get("logistic_regression", {}),
        )
        for head, estimator in self.single_heads.items():
            estimator.fit(
                matrix,
                [str(label[head]) for label in labels],
                [float(weight[head]) for weight in field_weights],
                scalar_logistic_config,
            )
        reasoning_order = self.class_order["reasoning_tags"]
        reasoning_weights = [
            float(weight["reasoning_tags"]) for weight in field_weights
        ]
        for tag in reasoning_order:
            self.reasoning_heads[tag].fit(
                matrix,
                [int(tag in label["reasoning_tags"]) for label in labels],
                reasoning_weights,
                reasoning_logistic_config,
            )
        self.fitted = True
        return self

    def transform(self, texts: Sequence[str]) -> sparse.spmatrix:
        if not self.fitted:
            raise ContractError("MODEL_NOT_FITTED", "model is not fitted")
        return self.vectorizer.transform(texts)

    def predict_probabilities(self, texts: Sequence[str]) -> dict[str, np.ndarray]:
        matrix = self.transform(texts)
        result = {
            head: estimator.predict_proba(matrix)
            for head, estimator in self.single_heads.items()
        }
        result["reasoning_tags"] = np.column_stack(
            [
                self.reasoning_heads[tag].predict_positive_probability(matrix)
                for tag in self.class_order["reasoning_tags"]
            ]
        )
        return result

    def feature_count(self) -> int:
        if not self.fitted:
            raise ContractError("MODEL_NOT_FITTED", "model is not fitted")
        return len(self.vectorizer.get_feature_names_out())

    def feature_counts(self) -> dict[str, int]:
        if not self.fitted:
            raise ContractError("MODEL_NOT_FITTED", "model is not fitted")
        counts = {
            name: len(transformer.vocabulary_)
            for name, transformer in self.vectorizer.transformer_list
        }
        return {
            "char": int(counts["char"]),
            "word": int(counts["word"]),
            "total": int(sum(counts.values())),
        }

    def convergence_diagnostics(self) -> dict[str, Any]:
        def describe(estimator: LogisticRegression | None, constant: Any) -> dict[str, Any]:
            if estimator is None:
                return {"constant": constant, "converged": True, "n_iter": []}
            n_iter = [int(value) for value in estimator.n_iter_]
            max_iter = int(estimator.get_params()["max_iter"])
            return {
                "constant": None,
                "solver": str(estimator.get_params()["solver"]),
                "max_iter": max_iter,
                "n_iter": n_iter,
                "converged": all(value < max_iter for value in n_iter),
            }

        return {
            "single_label": {
                head: describe(wrapper.estimator, wrapper.constant_label)
                for head, wrapper in self.single_heads.items()
            },
            "reasoning_tags": {
                tag: describe(wrapper.estimator, wrapper.constant)
                for tag, wrapper in self.reasoning_heads.items()
            },
        }

    def assert_contract(self) -> None:
        if set(self.class_order) != set(V1_HEADS):
            raise ContractError(
                "MODEL_CLASS_ORDER_INVALID", "model does not contain exactly seven heads"
            )
