import hashlib
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from case_base import create_afcba_case_base_from_preprocessed_data
from evaluation import evaluate_classifier_performance
from preprocessing import DataPreprocessor, PreprocessingConfig


class BaseDataProcessor:
    """Base class implementing common DataProcessor functionality."""

    def __init__(self, config_path: str | Path, data_path: str | Path):
        """Initialise base processor with configuration and data paths."""
        self.config_path = Path(config_path)
        self.data_path = Path(data_path)
        self.config = self._load_preprocessing_config(config_path)

        self._cached_X = None
        self._cached_y = None

    def _load_preprocessing_config(
        self, config_path: str | Path
    ) -> PreprocessingConfig:
        """Single point for loading YAML configs."""
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f)
        return PreprocessingConfig(**yaml_config)

    def get_dataset_name(self) -> str:
        """Return the name of the dataset. Must be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement get_dataset_name()")

    def get_cache_key_components(self, holdout_fraction: float) -> str:
        """Generate dataset-specific cache key components."""
        config_hash = hashlib.md5(
            str(self.config).encode() + str(holdout_fraction).encode()
        ).hexdigest()[:8]
        data_hash = hashlib.md5(str(self.data_path).encode()).hexdigest()[:8]
        dataset_name = self.get_dataset_name().lower()
        return f"{dataset_name}_experiment_{config_hash}_{data_hash}"

    def prepare_data(
        self, holdout_fraction: float, random_state: int, make_consistent=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for training and testing."""
        if self._cached_X is None or self._cached_y is None:
            print(f"Preprocessing {self.get_dataset_name()} data...")
            preprocessor = DataPreprocessor(config_path=str(self.config_path))
            self._cached_X, self._cached_y = preprocessor.preprocess_data(
                str(self.data_path)
            )
            print(
                f"✓ Preprocessed data: {len(self._cached_X)} samples, {len(self._cached_X.columns)} features"
            )

        X_train, X_test, y_train, y_test = train_test_split(
            self._cached_X,
            self._cached_y,
            test_size=holdout_fraction,
            stratify=self._cached_y,
            random_state=random_state,
        )

        return X_train, X_test, y_train, y_test

    def get_numerical_features(self) -> list[str]:
        """Return numerical features for scaling."""
        return self.config.features["numerical"]


class GTDProcessor(BaseDataProcessor):
    """GTD-specific data processor."""

    def __init__(
        self,
        config_path: str | Path = "config/GTD.yaml",
        data_path: str | Path = "data/gtd.xlsx",
    ):
        """Initialise GTD processor with configuration and data paths."""
        super().__init__(config_path, data_path)

    def get_dataset_name(self) -> str:
        """Return the name of the dataset."""
        return "GTD"


class ChurnProcessor(BaseDataProcessor):
    """Churn-specific data processor for telecom customer churn analysis."""

    def __init__(
        self,
        config_path: str | Path = "config/churn.yaml",
        data_path: str | Path = "data/churn.csv",
    ):
        """Initialise Churn processor with configuration and data paths."""
        super().__init__(config_path, data_path)

    def get_dataset_name(self) -> str:
        """Return the name of the dataset."""
        return "Churn"

    def prepare_data(
        self, holdout_fraction: float, random_state: int, make_consistent=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare Churn data."""
        if self._cached_X is None or self._cached_y is None:
            print(f"Preprocessing {self.get_dataset_name()} data...")

            df = pd.read_csv(self.data_path)
            if make_consistent:
                from evaluate_data import dataset_consistency_and_balance

                indices = dataset_consistency_and_balance([self])
                df = df.drop(df.index[indices])

            df["TotalCharges"].replace(" ", np.nan, inplace=True)
            df.dropna(subset=["TotalCharges"], inplace=True)
            df["TotalCharges"] = df["TotalCharges"].astype(float)

            df["Partner"] = np.where(df["Partner"] == "Yes", 1, 0)
            df["gender"] = np.where(df["gender"] == "Male", 1, 0)
            df["Dependents"] = np.where(df["Dependents"] == "Yes", 1, 0)
            df["PhoneService"] = np.where(df["PhoneService"] == "Yes", 1, 0)
            df["PaperlessBilling"] = np.where(df["PaperlessBilling"] == "Yes", 1, 0)
            df["Churn"] = np.where(df["Churn"] == "Yes", 1, 0)

            temp_path = self.data_path.parent / "temp_churn.csv"
            df.to_csv(temp_path, index=False)

            try:
                preprocessor = DataPreprocessor(config_path=str(self.config_path))
                self._cached_X, self._cached_y = preprocessor.preprocess_data(
                    str(temp_path)
                )
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

            print(
                f"✓ Preprocessed data: {len(self._cached_X)} samples, {len(self._cached_X.columns)} features"
            )

        if not holdout_fraction:
            return self._cached_X, None, self._cached_y, None

        X_train, X_test, y_train, y_test = train_test_split(
            self._cached_X,
            self._cached_y,
            test_size=holdout_fraction,
            stratify=self._cached_y,
            random_state=random_state,
        )

        return X_train, X_test, y_train, y_test


class AdmissionProcessor(BaseDataProcessor):
    """Admission-specific data processor with target rounding."""

    def __init__(
        self,
        config_path: str | Path = "config/admission.yaml",
        data_path: str | Path = "data/admission.csv",
    ):
        """Initialise Admission processor with configuration and data paths."""
        super().__init__(config_path, data_path)

    def get_dataset_name(self) -> str:
        """Return the name of the dataset."""
        return "Admission"

    def prepare_data(
        self, holdout_fraction: float, random_state: int, make_consistent=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare Admission data with target rounding before preprocessing."""
        if self._cached_X is None or self._cached_y is None:
            print(f"Preprocessing {self.get_dataset_name()} data...")

            df = pd.read_csv(self.data_path)
            if make_consistent:
                from evaluate_data import dataset_consistency_and_balance

                indices = dataset_consistency_and_balance([self])
                df = df.drop(df.index[indices])

            if "Chance of Admit " in df.columns:
                df["Chance of Admit "] = df["Chance of Admit "].round(0).astype(int)

            temp_path = self.data_path.parent / "temp_admission.csv"
            df.to_csv(temp_path, index=False)

            try:
                preprocessor = DataPreprocessor(config_path=str(self.config_path))
                self._cached_X, self._cached_y = preprocessor.preprocess_data(
                    str(temp_path)
                )
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

            print(
                f"✓ Preprocessed data: {len(self._cached_X)} samples, {len(self._cached_X.columns)} features"
            )

        if not holdout_fraction:
            return self._cached_X, None, self._cached_y, None

        X_train, X_test, y_train, y_test = train_test_split(
            self._cached_X,
            self._cached_y,
            test_size=holdout_fraction,
            stratify=self._cached_y,
            random_state=random_state,
        )
        print("Training class distribution:", y_train.value_counts())
        print("Test class distribution:", y_test.value_counts())

        return X_train, X_test, y_train, y_test


class COMPASProcessor(BaseDataProcessor):
    """COMPAS-specific data processor for recidivism prediction."""

    def __init__(
        self,
        config_path: str | Path = "config/compas.yaml",
        data_path: str | Path = "data/compas.csv",
    ):
        """Initialise COMPAS processor with configuration and data paths."""
        super().__init__(config_path, data_path)

    def get_dataset_name(self) -> str:
        """Return the name of the dataset."""
        return "COMPAS"

    def prepare_data(
        self, holdout_fraction: float, random_state: int, make_consistent=False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare COMPAS data with specific datetime processing."""
        if self._cached_X is None or self._cached_y is None:
            print(f"Preprocessing {self.get_dataset_name()} data...")

            df = pd.read_csv(self.data_path)

            if make_consistent:
                from evaluate_data import dataset_consistency_and_balance

                indices = dataset_consistency_and_balance([self])
                df = df.drop(df.index[indices])

            # # COMPAS-specific preprocessing
            # print("Processing jail dates and calculating jail time...")

            # # Convert jail dates to datetime
            # df["c_jail_in"] = pd.to_datetime(df["c_jail_in"], errors="coerce")
            # df.dropna(subset=["c_jail_in"], inplace=True)

            # df["c_jail_out"] = pd.to_datetime(df["c_jail_out"], errors="coerce")
            # df.dropna(subset=["c_jail_out"], inplace=True)

            # # Calculate jail time in days
            # df["jailtime"] = (
            #     ((df["c_jail_out"] - df["c_jail_in"]) / np.timedelta64(1, "D"))
            #     .round(0)
            #     .astype(int)
            # )

            # # Remove negative jail times (data quality issue)
            # df = df[df["jailtime"] >= 0]

            # # Drop original jail date columns
            # df.drop(columns=["c_jail_in", "c_jail_out"], inplace=True)

            # Save to temporary file for preprocessing
            temp_path = self.data_path.parent / "temp_compas.csv"
            df.to_csv(temp_path, index=False)

            try:
                preprocessor = DataPreprocessor(config_path=str(self.config_path))
                self._cached_X, self._cached_y = preprocessor.preprocess_data(
                    str(temp_path)
                )
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

            print(
                f"✓ Preprocessed data: {len(self._cached_X)} samples, {len(self._cached_X.columns)} features"
            )

        if not holdout_fraction:
            return self._cached_X, None, self._cached_y, None

        X_train, X_test, y_train, y_test = train_test_split(
            self._cached_X,
            self._cached_y,
            test_size=holdout_fraction,
            stratify=self._cached_y,
            random_state=random_state,
        )

        print("Training class distribution:", y_train.value_counts())
        print("Test class distribution:", y_test.value_counts())

        return X_train, X_test, y_train, y_test


class Experiment:
    """Generic experiment runner with unified evaluation and caching."""

    def __init__(
        self,
        processor: BaseDataProcessor,
        random_state: int = 42,
        cache_dir: str | Path = "cache",
        force_recache: bool = False,
    ):
        """Initialise experiment with data processor and configuration."""
        self.processor = processor
        self.random_state = random_state
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.force_recache = force_recache

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Store experiment configuration
        self.use_harmonic_authoritativeness = False

        self.results = {"sklearn_cv": {}, "afcba_cv": {}, "holdout_test": {}}

    def _generate_cache_key(self, holdout_fraction: float) -> str:
        """Generate cache key from processor and experiment params."""
        base_key = self.processor.get_cache_key_components(holdout_fraction)
        return f"{base_key}_{self.random_state}"

    def prepare_data(
        self, holdout_fraction: float = 0.2, make_consistent=False
    ) -> None:
        """Prepare and cache training/test data."""
        cache_key = self._generate_cache_key(holdout_fraction)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists() and not self.force_recache:
            print(f"Loading cached data from {cache_file}")
            with open(cache_file, "rb") as f:
                cached_data = pickle.load(f)
                self.X_train = cached_data["X_train"]
                self.X_test = cached_data["X_test"]
                self.y_train = cached_data["y_train"]
                self.y_test = cached_data["y_test"]
        else:
            print("Preparing fresh data...")
            self.X_train, self.X_test, self.y_train, self.y_test = (
                self.processor.prepare_data(
                    holdout_fraction, self.random_state, make_consistent
                )
            )

            self._export_dataset_metrics()

            # Cache the prepared data
            cached_data = {
                "X_train": self.X_train,
                "X_test": self.X_test,
                "y_train": self.y_train,
                "y_test": self.y_test,
            }
            with open(cache_file, "wb") as f:
                pickle.dump(cached_data, f)
            print(f"Data cached to {cache_file}")

    def _export_dataset_metrics(self):
        """Export simple dataset metrics to a dataset.json file."""
        from collections import Counter
        from datetime import datetime

        # Combine full dataset for metrics
        if self.y_test is not None:
            full_y = pd.concat([self.y_train, self.y_test])
            full_X = pd.concat([self.X_train, self.X_test])
        else:
            full_y = self.y_train
            full_X = self.X_train

        # Simple metrics
        class_distribution = dict(Counter(full_y))

        dataset_info = {
            "dataset_name": self.processor.get_dataset_name(),
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(full_X),
            "n_features": len(full_X.columns),
            "class_distribution": class_distribution,
            "n_classes": len(class_distribution),
        }

        # Export to dataset-specific JSON file
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        dataset_filename = f"dataset_{self.processor.get_dataset_name().lower()}.json"
        output_path = output_dir / dataset_filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        print(f"📊 Dataset metrics exported to: {output_path}")
        print(f"   Rows: {dataset_info['n_rows']:,}, Classes: {class_distribution}")

    def evaluate_holdout_test_set(self) -> None:
        """Evaluate models on holdout test set using the same logic as cross-validation."""
        if any(
            data is None
            for data in [self.X_train, self.X_test, self.y_train, self.y_test]
        ):
            raise ValueError("Data not prepared. Call prepare_data() first.")

        print("Evaluating on holdout test set...")

        # Prepare fold data in the same format as cross-validation
        fold_data = (self.X_train, self.X_test, self.y_train, self.y_test)
        numerical_features = self.processor.get_numerical_features()

        # Use the same evaluation logic as cross-validation
        result = self._evaluate_single_fold(
            fold_data,
            numerical_features,
            self.random_state,
            self.use_harmonic_authoritativeness,
        )

        # Store results in the holdout_test section
        for model_name, metrics in result["sklearn"].items():
            self.results["holdout_test"][model_name] = metrics

        for model_name, metrics in result["afcba"].items():
            self.results["holdout_test"][model_name] = metrics

        print("✓ Holdout evaluation completed")

    @staticmethod
    def _evaluate_single_fold(
        fold_data,
        numerical_features,
        random_state,
        use_harmonic_authoritativeness=False,
    ):
        """Evaluate a single cross-validation fold independently.
        This method is static to enable proper multiprocessing.

        Args:
            fold_data: Tuple of (X_train_fold, X_val_fold, y_train_fold, y_val_fold)
            numerical_features: List of numerical feature names for scaling
            random_state: Random state for reproducibility
            use_harmonic_authoritativeness: Whether to use harmonic authoritativeness in AF-CBA
        """
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = fold_data

        # Scale numerical features
        if numerical_features:
            scaler = StandardScaler()

            X_train_scaled = X_train_fold.copy()
            X_val_scaled = X_val_fold.copy()

            X_train_scaled[numerical_features] = scaler.fit_transform(
                X_train_fold[numerical_features]
            )
            X_val_scaled[numerical_features] = scaler.transform(
                X_val_fold[numerical_features]
            )
        else:
            X_train_scaled = X_train_fold
            X_val_scaled = X_val_fold

        # Evaluate sklearn models
        models = {
            "DecisionTree": DecisionTreeClassifier(random_state=random_state),
            "RandomForest": RandomForestClassifier(random_state=random_state),
        }

        sklearn_results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train_fold)
            y_pred = model.predict(X_val_scaled)

            sklearn_results[name] = {
                "accuracy": accuracy_score(y_val_fold, y_pred),
                "f1_score": f1_score(y_val_fold, y_pred),
                "mcc": matthews_corrcoef(y_val_fold, y_pred),
            }

        # Evaluate AF-CBA model
        test_cases = X_val_scaled.copy()
        true_outcomes = y_val_fold.copy()

        afcba_results = {}
        model_name = "AF-CBA-Harmonic" if use_harmonic_authoritativeness else "AF-CBA"

        case_base = create_afcba_case_base_from_preprocessed_data(
            X_train_scaled,
            y_train_fold,
            use_harmonic_authoritativeness=use_harmonic_authoritativeness,
            beta=1.0,
        )

        outcome_counts = case_base.cases[case_base.target_column].value_counts()
        default_outcome = outcome_counts.idxmax()
        non_default_outcome = next(
            o for o in outcome_counts.index if o != default_outcome
        )

        performance = evaluate_classifier_performance(
            case_base,
            test_cases,
            true_outcomes,
            default_outcome=default_outcome,
            non_default_outcome=non_default_outcome,
        )

        afcba_results[model_name] = {
            "accuracy": performance.get("accuracy"),
            "f1_score": performance.get("f1_score"),
            "mcc": performance.get("mcc"),
            "justified_case_rate": performance.get("justified_case_rate"),
        }

        return {"sklearn": sklearn_results, "afcba": afcba_results}

    def cross_validate_approaches(
        self, cv_folds: int = 5, n_jobs: int = -1, parallel_afcba: bool = False
    ) -> dict[str, Any]:
        """Parallel cross-validation for both sklearn and AF-CBA approaches.

        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all available cores)
            parallel_afcba: Whether to run AF-CBA approaches in parallel within each fold
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not prepared. Call prepare_data() first.")

        # Determine number of jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs <= 0:
            n_jobs = 1

        afcba_mode = "nested parallel" if parallel_afcba else "sequential"
        print(
            f"Cross-validation with {cv_folds} folds using {n_jobs} parallel jobs (AF-CBA: {afcba_mode})..."
        )

        X_full = pd.concat([self.X_train, self.X_test], axis=0).reset_index(drop=True)
        y_full = pd.concat([self.y_train, self.y_test], axis=0).reset_index(drop=True)

        skf = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=self.random_state
        )

        # Prepare fold data for parallel processing
        fold_data_list = []
        for train_idx, val_idx in skf.split(X_full, y_full):
            X_train_fold = X_full.iloc[train_idx]
            X_val_fold = X_full.iloc[val_idx]
            y_train_fold = y_full.iloc[train_idx]
            y_val_fold = y_full.iloc[val_idx]
            fold_data_list.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

        # Get numerical features for scaling
        numerical_features = self.processor.get_numerical_features()

        # Create wrapper function that includes harmonic authoritativeness setting
        def evaluate_fold_with_config(fold_data):
            return self._evaluate_single_fold(
                fold_data,
                numerical_features,
                self.random_state,
                self.use_harmonic_authoritativeness,
            )

        # Run parallel cross-validation
        print("Starting parallel fold evaluation...")
        try:
            if n_jobs == 1:
                # Sequential processing for debugging or single-core scenarios
                fold_results_list = []
                for i, fold_data in enumerate(
                    tqdm(fold_data_list, desc="Processing folds")
                ):
                    result = evaluate_fold_with_config(fold_data)
                    fold_results_list.append(result)
            else:
                # Parallel processing
                fold_results_list = Parallel(n_jobs=n_jobs, verbose=1)(
                    delayed(evaluate_fold_with_config)(fold_data)
                    for fold_data in fold_data_list
                )
        except Exception as e:
            raise ValueError('f"Parallel processing failed') from e

        # Aggregate results from all folds
        fold_results = {"sklearn": {}, "afcba": {}}

        # Initialize results structure
        for model_name in ["DecisionTree", "RandomForest"]:
            fold_results["sklearn"][model_name] = {
                "accuracy": [],
                "f1_score": [],
                "mcc": [],
            }

        # Initialize based on harmonic authoritativeness setting
        afcba_model_name = (
            "AF-CBA-Harmonic" if self.use_harmonic_authoritativeness else "AF-CBA"
        )
        fold_results["afcba"][afcba_model_name] = {
            "accuracy": [],
            "f1_score": [],
            "mcc": [],
            "justified_case_rate": [],
        }

        # Collect results from all folds
        for fold_result in fold_results_list:
            for model_name, metrics in fold_result["sklearn"].items():
                for metric_name, value in metrics.items():
                    fold_results["sklearn"][model_name][metric_name].append(value)

            for model_name, metrics in fold_result["afcba"].items():
                for metric_name, value in metrics.items():
                    fold_results["afcba"][model_name][metric_name].append(value)

        # Calculate statistics
        for category in ["sklearn", "afcba"]:
            self.results[f"{category}_cv"] = {}
            for model_name, metrics in fold_results[category].items():
                self.results[f"{category}_cv"][model_name] = {}
                for metric_name, values in metrics.items():
                    self.results[f"{category}_cv"][model_name][metric_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "values": values,
                    }

        print("✓ Cross-validation completed")

    def generate_results_table(self) -> pd.DataFrame:
        """Generate a comprehensive results table from all evaluations."""
        print("\nGenerating results table...")

        all_results = []

        # Cross-validation results
        for category in ["sklearn_cv", "afcba_cv"]:
            if category in self.results:
                for model_name, metrics in self.results[category].items():
                    for metric_name, stats in metrics.items():
                        if metric_name in [
                            "accuracy",
                            "f1_score",
                            "mcc",
                            "justified_case_rate",
                        ]:
                            all_results.append(
                                {
                                    "Model": model_name,
                                    "Evaluation": "Cross-Validation",
                                    "Metric": metric_name,
                                    "Score": stats["mean"],
                                    "Std": stats["std"],
                                }
                            )

        # Holdout test results
        if "holdout_test" in self.results:
            for model_name, metrics in self.results["holdout_test"].items():
                for metric_name, score in metrics.items():
                    if metric_name in [
                        "accuracy",
                        "f1_score",
                        "mcc",
                        "justified_case_rate",
                    ]:
                        all_results.append(
                            {
                                "Model": model_name,
                                "Evaluation": "Holdout Test",
                                "Metric": metric_name,
                                "Score": score,
                                "Std": 0.0,
                            }
                        )

        return pd.DataFrame(all_results)

    def run_complete_experiment(
        self,
        cv_folds: int = 5,
        holdout_fraction: float = 0.2,
        n_jobs: int = -1,
        use_parallel: bool = True,
        parallel_afcba: bool = False,
        use_harmonic_authoritativeness: bool = False,
    ) -> pd.DataFrame:
        """Run the complete experiment with comprehensive progress tracking.

        Args:
            cv_folds: Number of cross-validation folds
            holdout_fraction: Fraction of data to hold out for testing
            n_jobs: Number of parallel jobs for cross-validation
            use_parallel: Whether to use parallel cross-validation
            parallel_afcba: Whether to run AF-CBA approaches in parallel within each fold
            use_harmonic_authoritativeness: Whether to use harmonic authoritativeness in AF-CBA
        """
        start = time.time()
        dataset_name = self.processor.get_dataset_name()
        print(f"Starting Enhanced {dataset_name} Classification Experiment")
        print("=" * 60)

        # Store the harmonic authoritativeness setting for use throughout the experiment
        self.use_harmonic_authoritativeness = use_harmonic_authoritativeness

        print("Preparing and caching data...")
        self.prepare_data(holdout_fraction=holdout_fraction)
        print(
            f"✓ Data prepared: {len(self.X_train)} training, {len(self.X_test)} test cases"
        )

        cv_method = "parallel" if use_parallel else "sequential"
        afcba_method = "nested parallel" if parallel_afcba else "sequential"
        harmonic_status = (
            "with harmonic authoritativeness"
            if use_harmonic_authoritativeness
            else "standard"
        )
        print(
            f"\nStarting {cv_folds}-fold cross-validation ({cv_method}, AF-CBA: {afcba_method}, {harmonic_status})..."
        )

        if use_parallel:
            self.cross_validate_approaches(
                cv_folds=cv_folds, n_jobs=n_jobs, parallel_afcba=parallel_afcba
            )
        else:
            self.cross_validate_approaches_sequential(cv_folds=cv_folds)

        print("\nHold-out evaluation...")
        self.evaluate_holdout_test_set()

        print("\nGenerating comprehensive results...")
        results = self.generate_results_table()
        end = time.time()
        print(f"Experiment took {end-start} seconds.")
        return results

    def export_to_json(self, data, name, base_path="results"):
        """Export a JSON object to a file with an auto-incrementing iterator."""
        Path(base_path).mkdir(exist_ok=True)

        iterator = 1
        while True:
            filename = f"result_{name}_{iterator}.json"
            filepath = Path(base_path) / filename
            if not filepath.exists():
                break
            iterator += 1

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(filepath)


# Performance comparison functions with timing
def timed_experiment(experiment_func, *args, **kwargs):
    """Wrapper function to time experiments."""
    start_time = time.perf_counter()
    result = experiment_func(*args, **kwargs)
    end_time = time.perf_counter()
    duration = end_time - start_time
    return result, duration


def experimentGTD(
    use_parallel: bool = True,
    n_jobs: int = -1,
    parallel_afcba: bool = False,
    cv_folds: int = 5,
    use_harmonic_authoritativeness: bool = False,
    make_consistent=False,
):
    print("\n" + "=" * 60)
    print("GTD EXPERIMENT")
    print("=" * 60)
    processor = GTDProcessor()
    gtd_experiment = Experiment(processor=processor)
    gtd_results = gtd_experiment.run_complete_experiment(
        cv_folds=cv_folds,
        holdout_fraction=0.2,
        n_jobs=n_jobs,
        use_parallel=use_parallel,
        parallel_afcba=parallel_afcba,
        use_harmonic_authoritativeness=use_harmonic_authoritativeness,
    )
    gtd_export_path = gtd_experiment.export_to_json(
        gtd_results.to_dict("records"), name="GTD"
    )
    print(f"GTD results exported to: {gtd_export_path}")
    return gtd_results


def experimentAdmission(
    use_parallel: bool = True,
    n_jobs: int = -1,
    parallel_afcba: bool = False,
    cv_folds: int = 5,
    use_harmonic_authoritativeness: bool = False,
    make_consistent=False,
):
    print("\n" + "=" * 60)
    print("ADMISSION EXPERIMENT")
    print("=" * 60)
    processor = AdmissionProcessor()
    admission_experiment = Experiment(processor=processor)
    admission_results = admission_experiment.run_complete_experiment(
        cv_folds=cv_folds,
        holdout_fraction=0.2,
        n_jobs=n_jobs,
        use_parallel=use_parallel,
        parallel_afcba=parallel_afcba,
        use_harmonic_authoritativeness=use_harmonic_authoritativeness,
    )
    admission_export_path = admission_experiment.export_to_json(
        admission_results.to_dict("records"), name="Admission"
    )
    print(f"Admission results exported to: {admission_export_path}")
    return admission_results


def experimentChurn(
    use_parallel: bool = True,
    n_jobs: int = -1,
    parallel_afcba: bool = False,
    cv_folds: int = 5,
    use_harmonic_authoritativeness: bool = False,
    make_consistent=False,
):
    """Run comprehensive churn experiment with cross-validation and holdout evaluation."""
    print("\n" + "=" * 60)
    print("CHURN EXPERIMENT")
    print("=" * 60)
    processor = ChurnProcessor()
    churn_experiment = Experiment(processor=processor)
    churn_results = churn_experiment.run_complete_experiment(
        cv_folds=cv_folds,
        holdout_fraction=0.2,
        n_jobs=n_jobs,
        use_parallel=use_parallel,
        parallel_afcba=parallel_afcba,
        use_harmonic_authoritativeness=use_harmonic_authoritativeness,
    )
    churn_export_path = churn_experiment.export_to_json(
        churn_results.to_dict("records"), name="Churn"
    )
    print(f"Churn results exported to: {churn_export_path}")
    return churn_results


def experimentCOMPAS(
    use_parallel: bool = True,
    n_jobs: int = -1,
    parallel_afcba: bool = False,
    cv_folds: int = 5,
    use_harmonic_authoritativeness: bool = False,
    make_consistent: bool = False,
):
    """Run comprehensive COMPAS experiment with cross-validation and holdout evaluation."""
    print("\n" + "=" * 60)
    print("COMPAS EXPERIMENT")
    print("=" * 60)

    processor = COMPASProcessor()
    compas_experiment = Experiment(processor=processor)

    compas_results = compas_experiment.run_complete_experiment(
        cv_folds=cv_folds,
        holdout_fraction=0.2,
        n_jobs=n_jobs,
        use_parallel=use_parallel,
        parallel_afcba=parallel_afcba,
        use_harmonic_authoritativeness=use_harmonic_authoritativeness,
    )

    compas_export_path = compas_experiment.export_to_json(
        compas_results.to_dict("records"), name="COMPAS"
    )

    print(f"COMPAS results exported to: {compas_export_path}")
    return compas_results


def main():
    """Run experiments with different processors."""
    # Configuration parameters - adjust these to control experiment behaviour
    USE_PARALLEL = True
    CV_FOLDS = 5
    N_JOBS = 1
    N_JOBS = CV_FOLDS
    PARALLEL_AFCBA = False
    USE_HARMONIC_AUTHORITATIVENESS = False
    MAKE_CONSISTENT = False

    # experimentAdmission(
    #     use_parallel=USE_PARALLEL,
    #     n_jobs=N_JOBS,
    #     parallel_afcba=PARALLEL_AFCBA,
    #     cv_folds=CV_FOLDS,
    #     use_harmonic_authoritativeness=USE_HARMONIC_AUTHORITATIVENESS,
    #     make_consistent=MAKE_CONSISTENT,
    # )
    experimentGTD(
        use_parallel=USE_PARALLEL,
        n_jobs=N_JOBS,
        parallel_afcba=PARALLEL_AFCBA,
        cv_folds=CV_FOLDS,
        use_harmonic_authoritativeness=USE_HARMONIC_AUTHORITATIVENESS,
        make_consistent=MAKE_CONSISTENT,
    )
    experimentChurn(
        use_parallel=USE_PARALLEL,
        n_jobs=N_JOBS,
        parallel_afcba=PARALLEL_AFCBA,
        cv_folds=CV_FOLDS,
        use_harmonic_authoritativeness=USE_HARMONIC_AUTHORITATIVENESS,
        make_consistent=MAKE_CONSISTENT,
    )
    experimentCOMPAS(
        use_parallel=USE_PARALLEL,
        n_jobs=N_JOBS,
        parallel_afcba=PARALLEL_AFCBA,
        cv_folds=CV_FOLDS,
        use_harmonic_authoritativeness=USE_HARMONIC_AUTHORITATIVENESS,
        make_consistent=MAKE_CONSISTENT,
    )


if __name__ == "__main__":
    main()
