from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tqdm import tqdm

from af_cba import AFCBAGame
from case_base import CaseBase


def evaluate_classifier_performance(
    case_base: CaseBase,
    test_cases: pd.DataFrame,
    true_outcomes: pd.Series,
    default_outcome: Any,
    non_default_outcome: Any = None,
) -> dict[str, float]:
    """Evaluate AF-CBA classifier performance using fresh game instances.

    This function creates a fresh AFCBAGame for each test case to avoid
    memory accumulation issues.
    """
    if non_default_outcome is None:
        outcomes = case_base.cases[case_base.target_column].unique()
        non_default_outcome = next(o for o in outcomes if o != default_outcome)

    predictions = []
    default_count = 0
    total_cases = len(test_cases)

    print(f"Evaluating {total_cases} cases...")

    for i, (_, test_case) in tqdm(
        enumerate(test_cases.iterrows()),
        total=len(test_cases),
        desc="Evaluating AF-CBA",
        unit="case",
        mininterval=5,
    ):

        game = AFCBAGame(case_base)

        prediction, default_assigned = game.classify(
            test_case,
            default_outcome=default_outcome,
            non_default_outcome=non_default_outcome,
        )
        predictions.append(prediction)
        # Check if default was used
        if default_assigned:
            default_count += 1

    # Calculate metrics
    predictions = pd.Series(predictions)
    accuracy = accuracy_score(true_outcomes, predictions)
    f1 = f1_score(true_outcomes, predictions, average="macro", zero_division=0)
    mcc = matthews_corrcoef(true_outcomes, predictions)

    classified_cases = len(predictions)
    justified_cases = classified_cases - default_count
    justified_rate = justified_cases / total_cases

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "mcc": mcc,
        "justified_case_rate": justified_rate,
        "total_cases": total_cases,
        "classified_cases": classified_cases,
        "default_cases_assigned": default_count,
        "justified_cases": justified_cases,
    }
