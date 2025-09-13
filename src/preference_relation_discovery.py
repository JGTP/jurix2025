"""Optimized preference relation discovery for AF-CBA using TreeShap with absolute values.

This module extracts the sophisticated preference relation discovery algorithm
with all performance optimizations while using absolute SHAP values.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier


class PreferenceRelationDiscoverer:
    """Optimized preference relation discovery using TreeShap with absolute values."""

    def __init__(
        self,
        cases: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        dimension_info: dict[str, Any],
        min_correlation_diff: float = 0.1,
        max_preference_set_size: int = 5,
        max_correlation: float = 0.1,
        max_dimensions_to_consider: int = 8,
        max_relations_to_discover: int = 1000,
        allow_empty_compensation: bool = False,
        use_parallel_processing: bool = True,
        random_state: int = 42,
    ):
        """Initialise optimized preference relation discoverer.

        Args:
            cases: DataFrame containing the case data
            feature_columns: List of feature column names
            target_column: Name of the target column
            dimension_info: Dictionary mapping dimension names to DimensionInfo objects
            min_correlation_diff: Minimum relative difference in SHAP importance for preference relation
            max_preference_set_size: Maximum size of dimension sets to consider
            max_correlation: Maximum allowed correlation between dimensions in a set (epsilon threshold)
            max_dimensions_to_consider: Maximum number of top dimensions to consider
            max_relations_to_discover: Maximum number of preference relations to discover
            allow_empty_compensation: Whether to allow empty compensation sets
            use_parallel_processing: Whether to use parallel processing for large combination sets
            random_state: Random state for reproducibility
        """
        self.cases = cases
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.dimension_info = dimension_info
        self.min_correlation_diff = min_correlation_diff
        self.max_preference_set_size = max_preference_set_size
        self.max_correlation = max_correlation
        self.max_dimensions_to_consider = max_dimensions_to_consider
        self.max_relations_to_discover = max_relations_to_discover
        self.allow_empty_compensation = allow_empty_compensation
        self.use_parallel_processing = use_parallel_processing
        self.random_state = random_state

        # Caches for performance optimization
        self._cached_feature_importance = None
        self._cached_correlation_matrix = None

    def discover_preference_relations(
        self,
    ) -> dict[frozenset[str], list[frozenset[str]]]:
        """Optimized preference relation discovery using absolute SHAP values.

        Returns:
            Dictionary mapping worse dimension sets to lists of better dimension sets
            that can compensate for them.
        """
        print("Starting optimized preference relation discovery with absolute SHAP...")

        # Step 1: Calculate feature importance once using absolute SHAP
        feature_importance = self._calculate_feature_importance_once()

        # Step 2: Filter to most important dimensions only
        eligible_dims = [
            dim
            for dim in self.feature_columns
            if self.dimension_info[dim].direction != "neutral"
        ]

        # Sort by importance and take top N
        eligible_dims.sort(key=lambda d: feature_importance[d], reverse=True)
        eligible_dims = eligible_dims[: self.max_dimensions_to_consider]

        print(
            f"Considering top {len(eligible_dims)} most important dimensions out of "
            f"{len(self.feature_columns)} total"
        )
        print(f"Selected dimensions: {eligible_dims}")

        # Step 3: Cache correlations for eligible dimensions only
        correlation_cache = self._cache_dimension_correlations_optimised(eligible_dims)

        preference_relations = {}
        total_relations_found = 0

        # Step 4: Generate combinations incrementally with early termination
        for worse_size in range(1, len(self.feature_columns)):
            if total_relations_found >= self.max_relations_to_discover:
                print(
                    f"Reached maximum relations limit ({self.max_relations_to_discover}), stopping early"
                )
                break

            worse_combinations = list(combinations(eligible_dims, worse_size))
            print(
                f"Processing {len(worse_combinations)} combinations of size {worse_size}"
            )

            if self.use_parallel_processing and len(worse_combinations) > 10:
                # Step 5: Parallel processing for large combination sets
                batch_size = max(
                    1, len(worse_combinations) // 4
                )  # Split into 4 batches
                batches = [
                    worse_combinations[i : i + batch_size]
                    for i in range(0, len(worse_combinations), batch_size)
                ]

                params = {
                    "max_preference_set_size": self.max_preference_set_size,
                    "min_correlation_diff": self.min_correlation_diff,
                    "max_correlation": self.max_correlation,
                }

                # Prepare arguments for parallel processing
                batch_args = [
                    (
                        batch,
                        eligible_dims,
                        feature_importance,
                        correlation_cache,
                        params,
                    )
                    for batch in batches
                ]

                with ProcessPoolExecutor(max_workers=4) as executor:
                    future_to_batch = {
                        executor.submit(self._evaluate_combination_batch, args): i
                        for i, args in enumerate(batch_args)
                    }

                    for future in as_completed(future_to_batch):
                        batch_results = future.result()
                        for worse_set, better_sets in batch_results:
                            preference_relations[worse_set] = better_sets
                            total_relations_found += 1

                            if total_relations_found >= self.max_relations_to_discover:
                                print(
                                    "Reached maximum relations during parallel processing"
                                )
                                break

                        if total_relations_found >= self.max_relations_to_discover:
                            break

            else:
                # Sequential processing for smaller sets
                for worse_combo in worse_combinations:
                    if total_relations_found >= self.max_relations_to_discover:
                        break

                    worse_set = frozenset(worse_combo)
                    worse_importance = sum(
                        feature_importance[dim] for dim in worse_combo
                    )

                    better_sets = []

                    for better_size in range(1, self.max_preference_set_size + 1):
                        for better_combo in combinations(eligible_dims, better_size):
                            better_set = frozenset(better_combo)

                            # Skip if sets overlap
                            if worse_set & better_set:
                                continue

                            # Optimised correlation checking (Argumentation Scheme 3)
                            if self._dimensions_too_correlated_fast(
                                better_combo, correlation_cache
                            ):
                                continue

                            better_importance = sum(
                                feature_importance[dim] for dim in better_combo
                            )

                            # Use relative difference threshold
                            if worse_importance > 0:  # Avoid division by zero
                                relative_diff = (
                                    better_importance - worse_importance
                                ) / worse_importance
                                if relative_diff > self.min_correlation_diff:
                                    better_sets.append(better_set)

                    if better_sets:
                        # Sort by importance (strongest compensators first)
                        better_sets.sort(
                            key=lambda s: sum(feature_importance[dim] for dim in s),
                            reverse=True,
                        )
                        preference_relations[worse_set] = better_sets
                        total_relations_found += 1

        # Step 6: Add empty compensation if allowed
        if self.allow_empty_compensation:
            for worse_set in list(preference_relations.keys()):
                preference_relations[worse_set].append(frozenset())

        print(f"Discovered {len(preference_relations)} preference relations")
        print(
            f"Total compensating sets: {sum(len(v) for v in preference_relations.values())}"
        )

        return preference_relations

    def _calculate_feature_importance_once(self) -> dict[str, float]:
        """Calculate and cache absolute SHAP feature importance scores."""
        if self._cached_feature_importance is not None:
            return self._cached_feature_importance

        print("Calculating absolute SHAP feature importance scores...")

        X = self.cases[self.feature_columns].values
        y = self.cases[self.target_column].values
        classifier = HistGradientBoostingClassifier(
            random_state=self.random_state, max_iter=100
        )
        classifier.fit(X, y)
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X)

        shap_array = shap_values.values
        abs_shap = np.abs(shap_array)
        mean_abs_shap_values = abs_shap.mean(axis=0)
        self._cached_feature_importance = {
            name: float(importance)
            for name, importance in zip(
                self.feature_columns, mean_abs_shap_values, strict=False
            )
        }

        print(
            f"Calculated absolute SHAP importance for {len(self._cached_feature_importance)} features"
        )
        return self._cached_feature_importance

    def _cache_dimension_correlations_optimised(
        self, eligible_dims: list[str]
    ) -> dict[tuple[str, str], float]:
        """Cache pairwise correlations only for eligible dimensions."""
        if self._cached_correlation_matrix is not None:
            # Filter existing cache for eligible dimensions only
            return {
                (dim1, dim2): corr
                for (dim1, dim2), corr in self._cached_correlation_matrix.items()
                if dim1 in eligible_dims and dim2 in eligible_dims
            }

        print(f"Caching correlations for {len(eligible_dims)} eligible dimensions...")
        correlation_cache = {}

        # Only calculate correlations for eligible dimensions (much smaller matrix)
        for i, dim1 in enumerate(eligible_dims):
            for dim2 in eligible_dims[i + 1 :]:  # Only upper triangle, avoid duplicates
                correlation = self.cases[dim1].corr(self.cases[dim2])
                if not pd.isna(correlation):
                    abs_corr = abs(correlation)
                    # Store both directions for easy lookup
                    correlation_cache[(dim1, dim2)] = abs_corr
                    correlation_cache[(dim2, dim1)] = abs_corr
                else:
                    correlation_cache[(dim1, dim2)] = 0.0
                    correlation_cache[(dim2, dim1)] = 0.0

        self._cached_correlation_matrix = correlation_cache
        print(f"Cached {len(correlation_cache)} correlation pairs")
        return correlation_cache

    def _dimensions_too_correlated_fast(
        self,
        dimensions: tuple[str, ...],
        correlation_cache: dict[tuple[str, str], float],
    ) -> bool:
        """Optimised correlation checking with early exit.

        Implements Argumentation Scheme 3 (CORRELATION): if any pair of dimensions
        within the set exceeds max_correlation threshold, the preference relation
        cannot be said to hold.
        """
        # Check all pairs within the dimension set
        for i, dim1 in enumerate(dimensions):
            for dim2 in dimensions[i + 1 :]:
                correlation = correlation_cache.get((dim1, dim2), 0.0)
                if correlation > self.max_correlation:
                    return True  # Early exit on first violation
        return False

    def _evaluate_combination_batch(self, args: tuple) -> list[tuple]:
        """Evaluate a batch of combinations for parallel processing."""
        worse_combos, eligible_dims, feature_importance, correlation_cache, params = (
            args
        )

        results = []
        for worse_combo in worse_combos:
            worse_set = frozenset(worse_combo)
            worse_importance = sum(feature_importance[dim] for dim in worse_combo)

            better_sets = []
            for better_size in range(1, params["max_preference_set_size"] + 1):
                for better_combo in combinations(eligible_dims, better_size):
                    better_set = frozenset(better_combo)

                    # Skip if sets overlap
                    if worse_set & better_set:
                        continue

                    # Check correlation constraint
                    if self._dimensions_too_correlated_fast_static(
                        better_combo, correlation_cache, params["max_correlation"]
                    ):
                        continue

                    better_importance = sum(
                        feature_importance[dim] for dim in better_combo
                    )

                    # Use relative difference threshold
                    if worse_importance > 0:  # Avoid division by zero
                        relative_diff = (
                            better_importance - worse_importance
                        ) / worse_importance
                        if relative_diff > params["min_correlation_diff"]:
                            better_sets.append(better_set)

            if better_sets:
                # Sort by importance (strongest compensators first)
                better_sets.sort(
                    key=lambda s: sum(feature_importance[dim] for dim in s),
                    reverse=True,
                )
                results.append((worse_set, better_sets))

        return results

    @staticmethod
    def _dimensions_too_correlated_fast_static(
        dimensions: tuple[str, ...],
        correlation_cache: dict[tuple[str, str], float],
        max_correlation: float,
    ) -> bool:
        """Static version of correlation checking for parallel processing."""
        for i, dim1 in enumerate(dimensions):
            for dim2 in dimensions[i + 1 :]:
                correlation = correlation_cache.get((dim1, dim2), 0.0)
                if correlation > max_correlation:
                    return True
        return False

    def get_dimension_importance_summary(self) -> pd.DataFrame:
        """Get summary of individual dimension importances using absolute SHAP values."""
        feature_importance = self._calculate_feature_importance_once()

        summary_data = []

        for dim in self.feature_columns:
            dim_info = self.dimension_info[dim]
            correlation = self.cases[dim].corr(self.cases[self.target_column])
            abs_shap_importance = feature_importance[dim]

            summary_data.append(
                {
                    "dimension": dim,
                    "direction": dim_info.direction,
                    "coefficient": dim_info.coefficient,
                    "correlation_with_target": correlation,
                    "mean_abs_shap": abs_shap_importance,  # Primary importance measure
                    "mean_value": self.cases[dim].mean(),
                    "std_value": self.cases[dim].std(),
                    "is_neutral": dim_info.direction == "neutral",
                }
            )

        return pd.DataFrame(summary_data).sort_values("mean_abs_shap", ascending=False)

    def get_feature_importance_dict(self) -> dict[str, float]:
        """Get the cached feature importance dictionary."""
        return self._calculate_feature_importance_once()

    def clear_caches(self) -> None:
        """Clear all caches to free memory."""
        self._cached_feature_importance = None
        self._cached_correlation_matrix = None
