from __future__ import annotations

import json
from typing import Any

import pandas as pd
from tqdm import tqdm

from argumentation import ArgumentationFramework, ArgumentWrapper
from case_base import CaseBase
from game import GameOutcome, GroundedGame


class AFCBAArgument(ArgumentWrapper):
    """Base class for AF-CBA specific arguments."""

    pass


class Citation(AFCBAArgument):
    """Unified case citation - handles all case arguments regardless of strategic context."""

    def __init__(
        self,
        case: pd.Series,
        focus_case: pd.Series,
        case_name: str,
        introduced_by: str = None,  # "PRO" or "CON"
        responding_to: str = None,  # argument name this responds to
    ):
        # Determine strategic context for naming
        if responding_to and introduced_by == "CON":
            display_name = f"Counterexample({case_name}->{responding_to})"
            role = "counterexample"
        else:
            display_name = f"Citation({case_name})"
            role = "citation"

        content_data = {
            "type": "Citation",
            "case_name": case_name,
            "case_data": {
                k: v.item() if hasattr(v, "item") else v
                for k, v in case.to_dict().items()
            },
            "focus_data": focus_case.to_dict(),
            "strategic_context": {
                "introduced_by": introduced_by,
                "responding_to": responding_to,
                "role": role,
            },
        }

        super().__init__(
            name=display_name,
            content=json.dumps(content_data),
        )
        self.case = case
        self.focus_case = focus_case
        self.case_name = case_name
        self.strategic_context = content_data["strategic_context"]


class Worse(AFCBAArgument):
    """Distinguishing move - focus is worse on dimensions."""

    def __init__(
        self,
        precedent: pd.Series,
        focus_case: pd.Series,
        worse_dims: set[str],
        precedent_name: str,
    ):
        content_data = {
            "type": "Worse",
            "precedent_name": precedent_name,
            "precedent_data": {
                k: v.item() if hasattr(v, "item") else v
                for k, v in precedent.to_dict().items()
            },
            "focus_data": focus_case.to_dict(),
            "worse_dimensions": list(worse_dims),
        }

        dims_str = ",".join(sorted(worse_dims))
        super().__init__(
            name=f"Worse({precedent_name},[{dims_str}])",
            content=json.dumps(content_data),
        )
        self.precedent = precedent
        self.focus_case = focus_case
        self.worse_dimensions = worse_dims
        self.precedent_name = precedent_name


class Compensates(AFCBAArgument):
    """Compensation move - better dimensions compensate for worse ones."""

    def __init__(
        self,
        precedent: pd.Series,
        focus_case: pd.Series,
        worse_dims: set[str],
        better_dims: set[str],
        precedent_name: str,
    ):
        content_data = {
            "type": "Compensates",
            "precedent_name": precedent_name,
            "precedent_data": {
                k: v.item() if hasattr(v, "item") else v
                for k, v in precedent.to_dict().items()
            },
            "focus_data": focus_case.to_dict(),
            "worse_dimensions": list(worse_dims),
            "better_dimensions": list(better_dims),
        }

        worse_str = ",".join(sorted(worse_dims))
        better_str = ",".join(sorted(better_dims)) if better_dims else "∅"
        super().__init__(
            name=f"Compensates({precedent_name},[{better_str}]for[{worse_str}])",
            content=json.dumps(content_data),
        )
        self.precedent = precedent
        self.focus_case = focus_case
        self.worse_dimensions = worse_dims
        self.better_dimensions = better_dims
        self.precedent_name = precedent_name


class Transformed(AFCBAArgument):
    """Transformation - precedent transformed to have no differences."""

    def __init__(
        self,
        original_name: str,
        transformation_series: Any,
        focus: pd.Series,
    ):

        def convert_numpy_types(obj):
            if hasattr(obj, "item"):
                return obj.item()
            return obj

        if transformation_series == []:
            pass
        else:
            pass

        content_data = {
            "type": "Transformed",
            "original_precedent_name": original_name,
            "transformed_dimensions": str(transformation_series),
            "focus_data": {
                k: convert_numpy_types(v) for k, v in focus.to_dict().items()
            },
        }

        super().__init__(
            name=f"Transformed({original_name})",
            content=json.dumps(content_data),
        )
        self.original_precedent_name = original_name
        self.transformed_dimensions = transformation_series
        self.focus_case = focus


class LazyCounterexampleGenerator:
    """Generates counterexamples lazily on demand for a specific citation."""

    def __init__(
        self,
        target_case: pd.Series,
        target_case_name: str,
        focus_case: pd.Series,
        outcome_index: dict,
        case_base,
    ):
        self.target_case = target_case
        self.target_case_name = target_case_name
        self.focus_case = focus_case
        self.outcome_index = outcome_index
        self.case_base = case_base
        self.tried_indices: set[int] = set()
        self.target_outcome = target_case[case_base.target_column]

    def get_next_counterexample(self) -> Citation | None:
        """Generate the next available counterexample, or None if exhausted."""
        for outcome, case_indices in self.outcome_index.items():
            if outcome != self.target_outcome:  # Must have opposite outcome
                for case_idx in case_indices:
                    if case_idx not in self.tried_indices:
                        attacking_case = self.case_base.cases.loc[case_idx]

                        # Check if this is a valid attack: D(B,f) ⊄ D(A,f)
                        if self._is_valid_case_attack(attacking_case):
                            self.tried_indices.add(case_idx)
                            attacking_case_name = f"case_{case_idx}"

                            return Citation(
                                attacking_case,
                                self.focus_case,
                                attacking_case_name,
                                introduced_by="CON",
                                responding_to=self.target_case_name,
                            )

        return None  # Exhausted all possibilities

    def _is_valid_case_attack(self, attacking_case: pd.Series) -> bool:
        """Check if attacking_case can attack target_case: D(target,f) ⊄ D(attacking,f)"""
        target_diffs = set(
            self.case_base.calculate_relevant_differences(
                self.target_case, self.focus_case
            )
        )
        attacking_diffs = set(
            self.case_base.calculate_relevant_differences(
                attacking_case, self.focus_case
            )
        )

        # D(B,f) ⊄ D(A,f) means target_diffs is NOT a subset of attacking_diffs
        return not (target_diffs <= attacking_diffs)

    def has_more_counterexamples(self) -> bool:
        """Check if there are more untried counterexamples available."""
        for outcome, case_indices in self.outcome_index.items():
            if outcome != self.target_outcome:
                for case_idx in case_indices:
                    if case_idx not in self.tried_indices:
                        attacking_case = self.case_base.cases.loc[case_idx]
                        if self._is_valid_case_attack(attacking_case):
                            return True
        return False


class LazyAFCBAFramework:
    """Lazily builds AF-CBA framework using ArgumentationFramework with on-demand compensation."""

    def __init__(self, case_base: CaseBase, focus_case: pd.Series):
        self.case_base = case_base
        self.focus_case = focus_case
        self._outcome_index = self._build_outcome_index()
        self._lazy_generators: dict[str, LazyCounterexampleGenerator] = {}

        focus_id = str(focus_case.name) if hasattr(focus_case, "name") else "focus"
        self.af = ArgumentationFramework(f"afcba_{focus_id}")

        self._generated_arguments = set()
        self._expanded_from = set()

        self._current_precedent = None

        self._outcome_index = self._build_outcome_index()

        self._distinguishing_cache = {}

    def _build_outcome_index(self) -> dict[Any, list[int]]:
        """Build index mapping outcomes to case indices for efficient counterexample lookup."""
        target_col = self.case_base.target_column
        index = {}

        for idx, case in self.case_base.cases.iterrows():
            outcome = case[target_col]
            if outcome not in index:
                index[outcome] = []
            index[outcome].append(idx)

        return index

    def _get_distinguishing_dimensions_cached(
        self, precedent: pd.Series, precedent_name: str
    ) -> tuple[list[str], list[str]]:
        """Get distinguishing dimensions with caching to avoid recalculation."""
        if precedent_name in self._distinguishing_cache:
            return self._distinguishing_cache[precedent_name]

        worse_dims, better_dims = self.case_base.get_distinguishing_dimensions(
            precedent, self.focus_case
        )

        self._distinguishing_cache[precedent_name] = (worse_dims, better_dims)

        return worse_dims, better_dims

    def get_attackers(self, argument: ArgumentWrapper) -> list[ArgumentWrapper]:
        """Lazily generate attackers for an argument."""
        if argument in self._expanded_from:
            return list(self.af.get_attackers(argument.name))

        attackers = self._generate_attackers(argument)

        for attacker in attackers:
            if attacker not in self._generated_arguments:
                self.af.add_argument(attacker.name, attacker.content)
                self._generated_arguments.add(attacker)

            self.af.add_defeat(attacker.name, argument.name)

        self._expanded_from.add(argument)
        return attackers

    def _generate_attackers(self, argument: ArgumentWrapper) -> list[ArgumentWrapper]:
        """Generate attackers with on-demand compensation checking."""
        attackers = []

        if not argument.content:
            return attackers

        try:
            content_data = json.loads(argument.content)
            arg_type = content_data.get("type")
        except (json.JSONDecodeError, KeyError):
            # Fallback to name-based detection
            if argument.name.startswith(("Citation(", "Counterexample(")):
                arg_type = "Citation"
            elif argument.name.startswith("Worse("):
                arg_type = "Worse"
            else:
                return attackers

        if arg_type == "Citation":
            case = pd.Series(content_data["case_data"])
            focus = pd.Series(content_data["focus_data"])
            case_name = content_data["case_name"]

            case_outcome = case[self.case_base.target_column]
            focus_outcome = focus[self.case_base.target_column]

            # 1. Generate Worse move (single move with all worse dimensions)
            if case_outcome == focus_outcome:
                worse_dims, better_dims = self._get_distinguishing_dimensions_cached(
                    case, case_name
                )
                if worse_dims:
                    attackers.append(Worse(case, focus, set(worse_dims), case_name))

            # 2. Set up lazy counterexample generation
            if argument.name not in self._lazy_generators:
                self._lazy_generators[argument.name] = LazyCounterexampleGenerator(
                    case, case_name, focus, self._outcome_index, self.case_base
                )

            # Generate first counterexample
            first_counterexample = self._lazy_generators[
                argument.name
            ].get_next_counterexample()
            if first_counterexample:
                attackers.append(first_counterexample)

            # 3. Generate transformation move (if possible) - ON-DEMAND CHECK
            can_transform, series = self.case_base.can_transform(case, focus)
            if can_transform:
                attackers.append(Transformed(case_name, series, focus))

        elif arg_type == "Worse":
            # Generate compensation moves using ON-DEMAND checking
            precedent = pd.Series(content_data["precedent_data"])
            focus = pd.Series(content_data["focus_data"])
            worse_dims = set(content_data["worse_dimensions"])
            precedent_name = content_data["precedent_name"]

            _, better_dims = self._get_distinguishing_dimensions_cached(
                precedent, precedent_name
            )

            # Find valid compensation using on-demand checking
            valid_compensation = self.case_base.find_valid_compensation(
                better_dims, worse_dims
            )

            if valid_compensation is not None:
                attackers.append(
                    Compensates(
                        precedent,
                        focus,
                        worse_dims,
                        valid_compensation,
                        precedent_name,
                    )
                )

        # Note: Compensates and Transformed moves have no attackers (terminal moves)

        return attackers

    def get_next_counterexample(
        self, defeated_argument: ArgumentWrapper
    ) -> ArgumentWrapper | None:
        """Get the next counterexample when the current one is defeated."""
        # Find the argument this counterexample was attacking
        if defeated_argument.content:
            try:
                content_data = json.loads(defeated_argument.content)
                if content_data.get("type") == "Citation":
                    strategic_context = content_data.get("strategic_context", {})
                    responding_to = strategic_context.get("responding_to")

                    if responding_to and responding_to in self._lazy_generators:
                        next_counterexample = self._lazy_generators[
                            responding_to
                        ].get_next_counterexample()
                        if next_counterexample:
                            # Add to framework
                            if next_counterexample not in self._generated_arguments:
                                self.af.add_argument(
                                    next_counterexample.name,
                                    next_counterexample.content,
                                )
                                self._generated_arguments.add(next_counterexample)

                            # Add defeat relationship
                            self.af.add_defeat(next_counterexample.name, responding_to)
                            return next_counterexample
            except (json.JSONDecodeError, KeyError):
                pass

        return None

    def has_more_counterexamples(self, target_argument_name: str) -> bool:
        """Check if there are more counterexamples available for the target argument."""
        if target_argument_name in self._lazy_generators:
            return self._lazy_generators[
                target_argument_name
            ].has_more_counterexamples()
        return False

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics for debugging and monitoring."""
        return {
            "distinguishing_cache_size": len(self._distinguishing_cache),
            "generated_arguments": len(self._generated_arguments),
            "expanded_from": len(self._expanded_from),
            "outcome_index_size": len(self._outcome_index),
            "cached_precedents": list(self._distinguishing_cache.keys()),
        }

    def clear_cache(self):
        """Clear ALL caches and reset framework state."""
        # Clear our own caches
        self._distinguishing_cache.clear()
        self._generated_arguments.clear()
        self._expanded_from.clear()

        # Clear the underlying argumentation framework
        self.af.clear()


class AFCBAGame:
    """Main AF-CBA game for justification and classification using on-demand compensation."""

    def __init__(self, case_base: CaseBase):
        """Initialise AF-CBA game with a case base."""
        self.case_base = case_base
        self.default_cases = []

    def find_justification(
        self, focus_case: pd.Series, predicted_outcome: Any
    ) -> Any | None:
        """Find a winning strategy for the predicted outcome."""
        focus_with_outcome = focus_case.copy()
        focus_with_outcome[self.case_base.target_column] = predicted_outcome
        print(f"Searching for a justification for: {focus_with_outcome}")

        lazy_framework = LazyAFCBAFramework(self.case_base, focus_with_outcome)

        game = LazyGroundedGame(lazy_framework, predicted_outcome)

        best_precedents = self.case_base.find_best_precedents(focus_case)
        outcome = game.play_game(best_precedents)

        if outcome == GameOutcome.PRO_WINS:
            return game.get_winning_strategy()

        raise ValueError("No justification was found!")

    def get_winning_strategy(self) -> list | None:
        """Get the winning strategy from the last classification or justification."""
        raise NotImplementedError
        if self.last_game is None:
            return None

        return self.last_game.get_winning_strategy()

    def predict_with_coverage(
        self, test_cases: pd.DataFrame, default_outcome, non_default_outcome
    ) -> tuple[pd.Series, float]:
        """Classify multiple cases and calculate coverage statistics."""
        if self.case_base.target_column in test_cases.columns:
            test_cases = test_cases.drop(columns=[self.case_base.target_column])

        predictions = []

        for idx, row in tqdm(
            test_cases.iterrows(),
            desc="Predicting cases...",
            total=len(test_cases),
            mininterval=10,
            unit="case",
        ):
            try:
                prediction = self.classify(
                    row,
                    default_outcome=default_outcome,
                    non_default_outcome=non_default_outcome,
                )
                predictions.append(prediction)
            except Exception as e:
                raise ValueError(f"Error classifying case {idx}: {e}") from e

        predictions_series = pd.Series(predictions, index=test_cases.index)
        successful_predictions = predictions_series.notna().sum()
        total_cases = len(predictions_series)
        coverage = successful_predictions / total_cases

        print(
            f"🎯 Classification coverage: {coverage:.2%} ({successful_predictions}/{total_cases})"
        )

        return predictions_series, coverage

    def classify(
        self,
        focus_case: pd.Series,
        default_outcome: Any = None,
        non_default_outcome: Any = None,
    ) -> Any:
        """Classify a focus case using grounded semantics with on-demand compensation."""
        default_justified = self.justify_outcome(focus_case, default_outcome)
        non_default_justified = self.justify_outcome(focus_case, non_default_outcome)

        if default_justified and not non_default_justified:
            return default_outcome, False
        elif non_default_justified and not default_justified:
            return non_default_outcome, False
        elif not default_justified and not non_default_justified:
            return default_outcome, True
        elif default_justified and non_default_justified:
            print(
                f"Framework inconsistency detected: both {default_outcome} and {non_default_outcome}"
                f"can be justified for the same focus case. This indicates problems with the "
                f"case base or preference relations."
            )
        else:
            raise RuntimeError("Both unassigned!")

    def justify_outcome(self, focus_case, outcome):
        """Check if an outcome can be justified for the focus case."""
        focus_with_outcome = focus_case.copy()
        focus_with_outcome[self.case_base.target_column] = outcome

        lazy_framework = LazyAFCBAFramework(self.case_base, focus_with_outcome)

        game = LazyGroundedGame(lazy_framework, outcome)

        best_precedents = self.case_base.find_best_precedents(focus_with_outcome)
        if len(best_precedents) > 0:
            winner = game.play_game(best_precedents)
        else:
            winner = GameOutcome.CON_WINS

        outcome_justified = winner == GameOutcome.PRO_WINS

        if winner not in [GameOutcome.PRO_WINS, GameOutcome.CON_WINS]:
            raise ValueError(
                f"Unexpected game outcome for default outcome: {winner}. "
                f"Expected either PRO_WINS or CON_WINS."
            )

        return outcome_justified

    def get_default_cases(self) -> list[dict]:
        """Get list of cases that received default outcomes."""
        if not hasattr(self, "default_cases"):
            return []
        return self.default_cases.copy()

    def clear_default_cases(self) -> None:
        """Clear the list of cases that received default outcomes."""
        self.default_cases = []

    def get_default_cases_count(self) -> int:
        """Get the count of cases that received default outcomes."""
        if not hasattr(self, "default_cases"):
            return 0
        return len(self.default_cases)


class LazyGroundedGame(GroundedGame):
    """Modified GroundedGame that uses lazy framework."""

    def __init__(self, lazy_framework: LazyAFCBAFramework, predicted_outcome):
        self.lazy_framework = lazy_framework
        super().__init__(
            lazy_framework.af, predicted_outcome, self.lazy_framework.focus_case
        )

    def _get_attackers(self, argument: ArgumentWrapper) -> list[ArgumentWrapper]:
        """Override to use lazy generation."""
        return self.lazy_framework.get_attackers(argument)

    def _get_replies_for_argument(
        self, argument: ArgumentWrapper
    ) -> list[ArgumentWrapper]:
        """Override to use lazy generation."""
        return self.lazy_framework.get_attackers(argument)
