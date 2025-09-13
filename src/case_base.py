"""Refactored case base implementation with on-demand preference checking.

This version removes the elaborate preference relation pre-computation and instead
checks compensation and transformation moves on-demand when needed.
"""

from __future__ import annotations

from enum import Enum
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel, Field
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


class DimensionInferenceMethod(Enum):
    """Methods for inferring dimension directions from data."""

    PEARSON_CORRELATION = "pearson_correlation"
    LOGISTIC_REGRESSION = "logistic_regression"


class DimensionInfo(BaseModel):
    """Metadata about a numerical dimension for AF-CBA reasoning."""

    name: str = Field(..., description="Feature/dimension name")
    direction: str = Field(
        description="Value ordering: 'positive', 'negative', or 'neutral'",
    )
    coefficient: float = Field(
        description="Statistical coefficient used for direction inference"
    )

    def is_at_least_as_good_as(
        self, value1: Any, value2: Any, for_outcome: Any
    ) -> bool:
        """Check if value1 ≥ value2 from perspective of achieving for_outcome."""
        if self.direction == "neutral":
            return True
        elif self.direction not in ["positive", "negative"]:
            raise ValueError(f"Invalid direction: {self.direction}")

        if self.direction == "positive":
            if for_outcome == 1:
                return value1 >= value2
            else:
                return value1 <= value2
        else:
            if for_outcome == 1:
                return value1 <= value2
            else:
                return value1 >= value2

    def is_better_than(self, value1: Any, value2: Any, for_outcome: Any) -> bool:
        """Check if value1 > value2 from perspective of achieving for_outcome."""
        if self.direction not in ["positive", "negative"]:
            raise ValueError(f"Invalid direction: {self.direction}")

        if self.direction == "positive":
            if for_outcome == 1:
                return value1 > value2
            else:
                return value1 < value2
        else:
            if for_outcome == 1:
                return value1 < value2
            else:
                return value1 > value2

    def is_worse_than(self, value1: Any, value2: Any, for_outcome: Any) -> bool:
        """Check if value1 < value2 from perspective of achieving for_outcome."""
        return self.is_better_than(value2, value1, for_outcome)


class CaseBase:
    """Case base for AF-CBA reasoning with on-demand preference checking."""

    def __init__(
        self,
        cases: pd.DataFrame,
        target_column: str,
        dimension_info: dict[str, DimensionInfo] | None = None,
        inference_method: DimensionInferenceMethod = DimensionInferenceMethod.LOGISTIC_REGRESSION,
        min_corr: float = 0.00,
        allow_empty_compensation: bool = False,
        importance_threshold: float = 0.0,  # Threshold for SHAP importance difference
        max_compensation_set_size: int = 3,  # Max size of compensating dimension sets to try
        use_harmonic_authoritativeness: bool = False,
        beta: float = 1.0,
        random_state: int = 42,
    ):
        """Initialise case base for AF-CBA reasoning with on-demand preference checking."""
        self.cases = cases.copy()
        self.target_column = target_column
        self.feature_columns = [col for col in cases.columns if col != target_column]

        # Store parameters
        self.inference_method = inference_method
        self.min_corr = min_corr
        self.allow_empty_compensation = allow_empty_compensation
        self.importance_threshold = importance_threshold
        self.max_compensation_set_size = max_compensation_set_size
        self.use_harmonic_authoritativeness = use_harmonic_authoritativeness
        self.beta = beta
        self.random_state = random_state

        # Infer dimension information if not provided
        if dimension_info is not None:
            self.dimension_info = dimension_info
        else:
            self.dimension_info = {}
            self._infer_all_dimension_directions()

        # Compute and cache SHAP importance values for on-demand compensation checking
        self._compute_shap_importance()

        # Compute authoritativeness values if requested
        self.alpha_values = {}
        if self.use_harmonic_authoritativeness:
            self._precompute_authoritativeness()

    def _infer_all_dimension_directions(self) -> None:
        """Infer directions for all dimensions using the specified method."""
        for col in self.feature_columns:
            direction, coefficient = self._infer_single_dimension_direction(col)
            self.dimension_info[col] = DimensionInfo(
                name=col, direction=direction, coefficient=coefficient
            )

    def _infer_single_dimension_direction(self, column: str) -> tuple[str, float]:
        """Infer direction for a single dimension using the specified method."""
        if self.inference_method == DimensionInferenceMethod.PEARSON_CORRELATION:
            return self._infer_pearson_correlation(column)
        elif self.inference_method == DimensionInferenceMethod.LOGISTIC_REGRESSION:
            return self._infer_logistic_regression(column)
        else:
            raise ValueError(f"Unknown inference method: {self.inference_method}")

    def _infer_pearson_correlation(self, column: str) -> tuple[str, float]:
        """Infer direction using Pearson correlation."""
        correlation = self.cases[column].corr(self.cases[self.target_column])

        if pd.isna(correlation) or abs(correlation) < self.min_corr:
            return "neutral", 0.0
        elif correlation > 0:
            return "positive", correlation
        else:
            return "negative", correlation

    def _infer_logistic_regression(self, column: str) -> tuple[str, float]:
        """Infer direction using logistic regression coefficient."""
        try:
            X = self.cases[[column]].values.reshape(-1, 1)
            y = self.cases[self.target_column].values

            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            lr.fit(X, y)

            coefficient = lr.coef_[0][0]

            if abs(coefficient) < self.min_corr:
                return "neutral", 0.0
            elif coefficient > 0:
                return "positive", coefficient
            else:
                return "negative", coefficient

        except Exception as e:
            print(f"⚠️ Error evaluating coefficient for {column}: {e}")
            return "neutral", 0.0

    def _compute_shap_importance(self) -> None:
        """Compute and cache SHAP importance values for all features."""
        print("Computing SHAP importance values for on-demand compensation...")

        X = self.cases[self.feature_columns].values
        y = self.cases[self.target_column].values

        classifier = HistGradientBoostingClassifier(
            random_state=self.random_state, max_iter=100
        )
        classifier.fit(X, y)

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer(X)

        # Use absolute SHAP values as importance scores
        shap_array = shap_values.values
        abs_shap = np.abs(shap_array)
        mean_abs_shap_values = abs_shap.mean(axis=0)

        self.shap_importance = {
            name: float(importance)
            for name, importance in zip(
                self.feature_columns, mean_abs_shap_values, strict=False
            )
        }

        print(f"✓ Computed SHAP importance for {len(self.shap_importance)} features")

    def can_compensate(self, better_dims: set[str], worse_dims: set[str]) -> bool:
        """Check if better_dims can compensate for worse_dims based on SHAP importance."""
        # if len(better_dims) == 0:
        #     return self.allow_empty_compensation

        # # Skip compensation check if either set contains neutral dimensions
        # for dim in better_dims | worse_dims:
        #     if (
        #         dim in self.dimension_info
        #         and self.dimension_info[dim].direction == "neutral"
        #     ):
        #         return False

        better_importance = sum(
            self.shap_importance.get(dim, 0.0) for dim in better_dims
        )
        worse_importance = sum(self.shap_importance.get(dim, 0.0) for dim in worse_dims)

        if (better_importance - worse_importance) >= self.importance_threshold:
            # print(rf" Preference relation: {worse_dims} \prec {better_dims}")
            return True

    def can_transform(
        self, precedent: pd.Series, focus: pd.Series
    ) -> tuple[bool, list]:
        """Check if precedent can be transformed to eliminate differences with focus.

        Transformation shows that the original citation (precedent) can have all its
        relevant differences with focus compensated away through a series of compensation
        moves, effectively "transforming" it into a case equivalent to focus.

        Returns:
            (can_transform: bool, compensation_moves: list of (better_set, worse_set) tuples)
        """
        # Get distinguishing dimensions:
        # worse_dims: dimensions where precedent is worse than focus
        # better_dims: dimensions where precedent is better than focus
        worse_dims, better_dims = self.get_distinguishing_dimensions(focus, precedent)

        if not worse_dims:
            return True, []  # Already no differences

        if not better_dims:
            return False, []  # No good dimensions to compensate with

        # Construct compensation series greedily
        return self._construct_compensation_series(set(better_dims), set(worse_dims))

    def _construct_compensation_series(
        self, better_dims: set[str], worse_dims: set[str]
    ) -> tuple[bool, list]:
        """Greedily construct compensation moves to cover all worse dimensions."""
        compensation_moves = []
        remaining_worse = worse_dims.copy()

        while remaining_worse:
            # Try full compensation first
            if self.can_compensate(better_dims, remaining_worse):
                compensation_moves.append(
                    (frozenset(better_dims), frozenset(remaining_worse))
                )
                break

            # Find the largest compensatable subset by reducing worse_dims
            compensated = False
            for worse_size in range(len(remaining_worse), 0, -1):
                if compensated:
                    break
                for worse_subset in combinations(remaining_worse, worse_size):
                    worse_set = set(worse_subset)

                    # Try full better_dims first, then subsets if needed
                    if self.can_compensate(better_dims, worse_set):
                        compensation_moves.append(
                            (frozenset(better_dims), frozenset(worse_set))
                        )
                        remaining_worse -= worse_set
                        compensated = True
                        break
                    else:
                        # Try subsets of better_dims
                        for better_size in range(len(better_dims), 0, -1):
                            for better_subset in combinations(better_dims, better_size):
                                better_set = set(better_subset)
                                if self.can_compensate(better_set, worse_set):
                                    compensation_moves.append(
                                        (frozenset(better_set), frozenset(worse_set))
                                    )
                                    remaining_worse -= worse_set
                                    compensated = True
                                    break
                            if compensated:
                                break
                    if compensated:
                        break

            if not compensated:
                return False, []  # Cannot compensate any remaining dimensions

        return True, compensation_moves

    def find_valid_compensation(
        self, better_dims: list[str], worse_dims: set[str]
    ) -> set[str] | None:
        """Find a valid compensation set from better_dims for worse_dims."""
        # Try empty compensation first if allowed
        # if self.allow_empty_compensation:
        #     if self.can_compensate(set(), worse_dims):
        #         return set()

        # Try combinations of better dimensions
        # for r in range(
        #     1, min(len(better_dims) + 1, self.max_compensation_set_size + 1)
        # ):
        #     for compensation_subset in combinations(better_dims, r):
        #         compensation_set = set(compensation_subset)
        #         if self.can_compensate(compensation_set, worse_dims):
        #             return compensation_set
        if self.can_compensate(better_dims, worse_dims):
            return better_dims
        return None

    def _precompute_authoritativeness(self) -> None:
        """Pre-compute authoritativeness scores for all cases."""
        print("Precomputing harmonic authoritativeness values...")
        for idx, case in tqdm(
            self.cases.iterrows(),
            total=len(self.cases),
            desc="Computing authoritativeness",
        ):
            self.alpha_values[idx] = self._calculate_harmonic_authoritativeness(case)

    def _calculate_harmonic_authoritativeness(self, case: pd.Series) -> float:
        """Calculate harmonic authoritativeness score for a case."""
        case_outcome = case[self.target_column]

        na = 0
        for idx, other_case in self.cases.iterrows():
            if other_case[self.target_column] == case_outcome:
                differences = self.calculate_relevant_differences(case, other_case)
                if len(differences) == 0:
                    na += 1

        nd = 0
        for idx, other_case in self.cases.iterrows():
            if other_case[self.target_column] != case_outcome:
                differences = self.calculate_relevant_differences(case, other_case)
                if len(differences) == 0:
                    nd += 1

        if na == 0:
            return 0.0

        if na + nd == 0:
            return 1.0

        relative_auth = na / (na + nd)
        absolute_auth = na / len(self.cases)

        beta_sq = self.beta**2
        numerator = (1 + beta_sq) * relative_auth * absolute_auth
        denominator = beta_sq * relative_auth + absolute_auth

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def calculate_relevant_differences(
        self, precedent_case: pd.Series, focus_case: pd.Series
    ) -> list[str]:
        """Calculate relevant differences D(c,f)."""
        relevant_differences = []
        focus_outcome = focus_case[self.target_column]

        for dim in self.feature_columns:
            if dim not in self.dimension_info:
                continue

            dim_info = self.dimension_info[dim]
            precedent_val = precedent_case[dim]
            focus_val = focus_case[dim]

            if not dim_info.is_at_least_as_good_as(
                precedent_val, focus_val, focus_outcome
            ):
                relevant_differences.append(dim)

        return relevant_differences

    def precedent_forces_focus(
        self, precedent_case: pd.Series, focus_case: pd.Series
    ) -> bool:
        """Check if precedent ≤_precedent_outcome focus."""
        precedent_outcome = precedent_case[self.target_column]

        for dim in self.feature_columns:
            if dim not in self.dimension_info:
                continue

            dim_info = self.dimension_info[dim]
            precedent_val = precedent_case[dim]
            focus_val = focus_case[dim]

            if not dim_info.is_at_least_as_good_as(
                focus_val, precedent_val, precedent_outcome
            ):
                return False

        return True

    def get_distinguishing_dimensions(
        self, precedent: pd.Series, focus_case: pd.Series
    ) -> tuple[list[str], list[str]]:
        """Find dimensions where focus is worse/better than precedent for focus outcome."""
        worse_dims = []
        better_dims = []
        focus_outcome = focus_case[self.target_column]

        for dim in self.feature_columns:
            if dim not in self.dimension_info:
                continue

            dim_info = self.dimension_info[dim]
            precedent_val = precedent[dim]
            focus_val = focus_case[dim]

            if precedent_val == focus_val:
                continue

            if dim_info.is_worse_than(focus_val, precedent_val, focus_outcome):
                worse_dims.append(dim)
            elif dim_info.is_better_than(focus_val, precedent_val, focus_outcome):
                better_dims.append(dim)
        return worse_dims, better_dims

    def find_best_precedents(self, focus_case: pd.Series) -> pd.DataFrame:
        """Find best precedents using ICAIL repository approach."""
        focus_outcome = focus_case[self.target_column]
        candidates = self.cases[self.cases[self.target_column] == focus_outcome].copy()

        if hasattr(focus_case, "name") and focus_case.name in candidates.index:
            candidates = candidates.drop(focus_case.name)

        if len(candidates) == 0:
            return pd.DataFrame()

        comparisons = []
        for idx, candidate in candidates.iterrows():
            differences = self.calculate_relevant_differences(candidate, focus_case)
            diff_set = set(differences) if differences else set()

            comparison = {
                "idx": idx,
                "rel_differences": diff_set,
            }

            if self.use_harmonic_authoritativeness:
                comparison["alpha"] = self.alpha_values[idx]

            comparisons.append(comparison)

        bested = set()

        for c in comparisons:
            if c["idx"] not in bested:
                if self.use_harmonic_authoritativeness:
                    bested.update(self._inner_loop_alpha(comparisons, c, bested))
                else:
                    bested.update(self._inner_loop_naive(comparisons, c, bested))

        # Return non-bested candidates
        best_indices = [c["idx"] for c in comparisons if c["idx"] not in bested]
        return candidates.loc[best_indices]

    def _inner_loop_naive(self, comparisons, c, already_bested):
        """ICAIL-style naive forcing check."""
        newly_bested = set()
        for oc in comparisons:
            if (
                oc["idx"] not in already_bested
                and c["rel_differences"] < oc["rel_differences"]
            ):  # c bests oc
                newly_bested.add(oc["idx"])
        return newly_bested

    def _inner_loop_alpha(self, comparisons, c, already_bested):
        """ICAIL-style alpha-aware forcing check."""
        newly_bested = set()
        for oc in comparisons:
            if (
                oc["idx"] not in already_bested
                and c["rel_differences"] < oc["rel_differences"]
                and c["alpha"] >= oc["alpha"]
            ):  # c bests oc
                newly_bested.add(oc["idx"])
        return newly_bested

    def is_forced_decision(
        self, focus_case: pd.Series, for_outcome: Any
    ) -> tuple[bool, pd.Series | None]:
        """Check if focus case decision is forced by precedential constraint."""
        candidates = self.cases[self.cases[self.target_column] == for_outcome]

        for idx, candidate in candidates.iterrows():
            if self.precedent_forces_focus(candidate, focus_case):
                return True, candidate

        return False, None

    def check_case_base_consistency(self) -> tuple[bool, list[tuple[int, int]]]:
        """Check case base consistency per AF-CBA Definition 3."""
        inconsistent_pairs = []
        case_indices = list(self.cases.iterrows())

        for idx1, case1 in tqdm(
            case_indices, desc="Checking consistency", mininterval=10
        ):
            for idx2, case2 in case_indices:
                if idx1 >= idx2:
                    continue
                outcome1 = case1[self.target_column]
                outcome2 = case2[self.target_column]

                if outcome1 == outcome2:
                    continue

                if self.precedent_forces_focus(case1, case2):
                    inconsistent_pairs.append((idx1, idx2))

        is_consistent = len(inconsistent_pairs) == 0
        return is_consistent, inconsistent_pairs

    def get_cases_with_outcome(self, outcome: Any) -> pd.DataFrame:
        """Get all cases with a specific outcome."""
        return self.cases[self.cases[self.target_column] == outcome].copy()

    def get_case_by_name(self, case_name: str) -> pd.Series | None:
        """Get a case by its name/index."""
        if case_name.startswith("case_"):
            try:
                idx = int(case_name.split("_")[1])
                return self.cases.iloc[idx] if idx < len(self.cases) else None
            except (ValueError, IndexError):
                return None
        else:
            try:
                idx = int(case_name)
                return self.cases.iloc[idx] if idx < len(self.cases) else None
            except (ValueError, IndexError):
                try:
                    return self.cases.loc[case_name]
                except KeyError:
                    return None

    def summarise_dimensions(self) -> pd.DataFrame:
        """Get a summary of all dimensions and their properties."""
        summary_data = []

        for dim in self.feature_columns:
            correlation = self.cases[dim].corr(self.cases[self.target_column])
            dim_info = self.dimension_info[dim]
            shap_importance = self.shap_importance.get(dim, 0.0)

            summary_data.append(
                {
                    "dimension": dim,
                    "direction": dim_info.direction,
                    "method": self.inference_method.value,
                    "coefficient": dim_info.coefficient,
                    "correlation_with_target": correlation,
                    "shap_importance": shap_importance,
                    "mean_value": self.cases[dim].mean(),
                    "std_value": self.cases[dim].std(),
                    "is_neutral": dim_info.direction == "neutral",
                }
            )

        return pd.DataFrame(summary_data).sort_values(
            "shap_importance", key=abs, ascending=False
        )

    def get_inconsistency(self) -> tuple[float, int]:
        """Get the percentage consistency and number of case removals needed."""
        inds = list(range(len(self.cases)))

        forcings = self._get_forcings(inds)
        inconsistent_forcings = self._determine_inconsistent_forcings(inds, forcings)

        removals = 0
        while sum(s := [len(inconsistent_forcings[i]) for i in inds]) != 0:
            k = s.index(max(s))

            for i in inds:
                inconsistent_forcings[i] -= {k}
            inconsistent_forcings[k] = set()
            removals += 1

        consistency = 100 * (1 - removals / len(self.cases))
        return consistency, removals

    def _get_forcings(self, inds: list[int]) -> set[tuple[int, int]]:
        """Get all forcing relations between cases."""
        forcings = set()

        for i in tqdm(inds, desc="Getting forcings...", mininterval=10):
            case_i = self.cases.iloc[i]
            for j in inds:
                case_j = self.cases.iloc[j]

                if self.precedent_forces_focus(case_i, case_j):
                    forcings.add((i, j))

        return forcings

    def _determine_inconsistent_forcings(
        self, inds: list[int], forcings: set[tuple[int, int]]
    ) -> dict[int, set[int]]:
        """Determine which forcings lead to inconsistencies."""
        inconsistent_forcings_pairs = {
            (i, j)
            for (i, j) in tqdm(
                forcings, desc="Getting inconsistencies...", mininterval=10
            )
            if self.cases.iloc[i][self.target_column]
            != self.cases.iloc[j][self.target_column]
        }

        inconsistent_dict = {i: set() for i in inds}
        for i, j in inconsistent_forcings_pairs:
            inconsistent_dict[i].add(j)
            inconsistent_dict[j].add(i)

        return inconsistent_dict

    def get_inconsistency_with_removals(self) -> tuple[float, int, list[int]]:
        """Get the percentage consistency, number of case removals needed, and the specific
        case indices that should be removed to achieve consistency.

        Returns:
            tuple: (consistency_percentage, number_of_removals, list_of_indices_to_remove)
        """
        inds = list(range(len(self.cases)))

        forcings = self._get_forcings(inds)
        inconsistent_forcings = self._determine_inconsistent_forcings(inds, forcings)

        removals = 0
        removed_indices = []

        while sum(s := [len(inconsistent_forcings[i]) for i in inds]) != 0:
            # Find the case involved in the most inconsistencies
            k = s.index(max(s))

            # Track which case is being removed
            removed_indices.append(k)

            # Remove this case from all inconsistency sets
            for i in inds:
                inconsistent_forcings[i] -= {k}
            inconsistent_forcings[k] = set()
            removals += 1

        consistency = 100 * (1 - removals / len(self.cases))
        return consistency, removals, removed_indices

    def get_detailed_inconsistency_report(self) -> dict:
        """Get a detailed report of inconsistency including which specific cases should be removed,
        their original indices, and summary statistics.

        Returns:
            dict: Detailed inconsistency analysis
        """
        consistency_percentage, n_removals, removal_indices = (
            self.get_inconsistency_with_removals()
        )

        # Get the actual case data for removed cases
        removed_cases = []
        if removal_indices:
            for idx in removal_indices:
                case_data = self.cases.iloc[idx].to_dict()
                case_data["original_index"] = idx
                case_data["original_dataframe_index"] = self.cases.index[idx]
                removed_cases.append(case_data)

        # Calculate some additional statistics
        total_cases = len(self.cases)
        removal_percentage = (n_removals / total_cases) * 100 if total_cases > 0 else 0

        report = {
            "consistency_analysis": {
                "total_cases": total_cases,
                "consistency_percentage": round(consistency_percentage, 2),
                "inconsistent_cases": n_removals,
                "removal_percentage": round(removal_percentage, 2),
            },
            "cases_to_remove": {
                "count": n_removals,
                "indices": removal_indices,
                "detailed_cases": removed_cases,
            },
            "summary": {
                "action_required": f"Remove {n_removals} cases ({removal_percentage:.1f}% of dataset) to achieve consistency",
                "resulting_dataset_size": total_cases - n_removals,
            },
        }

        return report

    def __len__(self) -> int:
        """Return number of cases in the case base."""
        return len(self.cases)

    def __repr__(self) -> str:
        """String representation of the case base."""
        return (
            f"CaseBase(cases={len(self.cases)}, "
            f"dimensions={len(self.feature_columns)}, "
            f"target='{self.target_column}', "
            f"method={self.inference_method.value}, "
            f"on_demand_compensation=True, "
            f"harmonic_auth={self.use_harmonic_authoritativeness})"
        )


def create_afcba_case_base_from_preprocessed_data(
    X: pd.DataFrame,
    y: pd.Series,
    dimension_info: dict[str, DimensionInfo] | None = None,
    inference_method: str = "logreg",
    min_corr: float = 0.00,
    allow_empty_compensation: bool = False,
    importance_threshold: float = 0.01,
    max_compensation_set_size: int = 5,
    use_harmonic_authoritativeness: bool = False,
    beta: float = 1.0,
    random_state: int = 42,
) -> CaseBase:
    """Create AF-CBA case base with on-demand preference checking."""
    cases = X.copy()
    cases["target"] = y

    if inference_method == "logreg":
        inference_method = DimensionInferenceMethod.LOGISTIC_REGRESSION
    elif inference_method == "pearson":
        inference_method = DimensionInferenceMethod.PEARSON_CORRELATION

    case_base = CaseBase(
        cases,
        target_column="target",
        dimension_info=dimension_info,
        inference_method=inference_method,
        min_corr=min_corr,
        allow_empty_compensation=allow_empty_compensation,
        importance_threshold=importance_threshold,
        max_compensation_set_size=max_compensation_set_size,
        use_harmonic_authoritativeness=use_harmonic_authoritativeness,
        beta=beta,
        random_state=random_state,
    )

    return case_base
