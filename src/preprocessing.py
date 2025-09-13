"""Enhanced preprocessing configuration supporting different categorical types.

Extends the existing preprocessing to properly handle:
- True categorical (no ordering): arbitrary category codes
- Ordinal (meaningful ordering): ranked categories
- Binary (from one-hot): presence/absence indicators
- Numerical (continuous): standard numeric features
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class PreprocessingConfig(BaseModel):
    """Enhanced configuration schema supporting different categorical types."""

    target: str = Field(..., description="Name of the target column")

    features: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "numerical": [],
            "categorical": [],
            "ordinal": [],
            "binary": [],
            "dates": [],
            "exclude": [],
        },
        description="Feature categorisation by type",
    )

    preprocessing: dict[str, Any] = Field(
        default_factory=dict, description="Preprocessing parameters"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Ensure all required feature categories exist."""
        required_keys = {
            "numerical",
            "categorical",
            "ordinal",
            "binary",
            "dates",
            "exclude",
        }
        for key in required_keys:
            if key not in v or v[key] is None:
                v[key] = []
        return v

    @model_validator(mode="after")
    def set_preprocessing_defaults(self):
        """Set default preprocessing parameters."""
        defaults = {
            "min_year": None,
            "year_column": "iyear",
            "sample_size": None,
            "missing_value_codes": {},
            "standardise_numerical": True,
            "one_hot_categoricals": True,
            "one_hot_ordinals": False,
        }
        for key, default_value in defaults.items():
            if key not in self.preprocessing:
                self.preprocessing[key] = default_value
        return self


class DataPreprocessor:
    """Enhanced preprocessor supporting different categorical types."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        config: PreprocessingConfig | None = None,
    ):
        """Initialise preprocessor with enhanced configuration."""
        if config_path is not None:
            self.config = self._load_config(config_path)
        elif config is not None:
            self.config = config
        else:
            raise ValueError("Must provide either config_path or config")

        self.label_encoder: LabelEncoder | None = None
        self.categorical_encoders: dict[str, OneHotEncoder] = {}
        self.ordinal_encoders: dict[str, OneHotEncoder] = {}
        self.scaler: StandardScaler | None = None
        self.target_type: str | None = None

    def _load_config(self, config_path: str | Path) -> PreprocessingConfig:
        """Load and validate enhanced configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

        try:
            return PreprocessingConfig(**raw_config)
        except Exception as e:
            raise ValueError(f"Invalid configuration structure: {e}")

    def _validate_column_coverage(self, df: pd.DataFrame) -> None:
        """Validate that all columns are explicitly accounted for in configuration."""
        all_columns = set(df.columns)
        all_columns.discard(self.config.target)

        included_columns = set()

        feature_categories = [
            "numerical",
            "categorical",
            "ordinal",
            "dates",
            "binary",
        ]

        for category in feature_categories:
            if category in self.config.features:
                included_columns.update(self.config.features[category])

        excluded_columns = set(self.config.features["exclude"])

        overlapping_columns = included_columns & excluded_columns
        if overlapping_columns:
            raise ValueError(
                f"Configuration error: The following columns are both included in feature "
                f"categories AND listed in the exclude list: {sorted(overlapping_columns)}\n"
                f"Please remove these columns from either the feature categories or the exclude list."
            )

        mentioned_columns = included_columns | excluded_columns

        unaccounted_columns = all_columns - mentioned_columns

        if unaccounted_columns:
            raise ValueError(
                f"Unaccounted columns found in dataset. All columns must be explicitly "
                f"categorised or excluded. Unaccounted columns: {sorted(unaccounted_columns)}\n"
                f"Please add these columns to one of the feature categories "
                f"({feature_categories}) or to the 'exclude' list in your configuration."
            )

        missing_columns = mentioned_columns - all_columns
        if missing_columns:
            print(
                f"⚠️  Warning: Configuration mentions columns not found in data: {sorted(missing_columns)}"
            )

    def preprocess_data(self, data_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
        """Preprocess data with enhanced categorical handling."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if data_path.suffix.lower() == ".xlsx":
            df = pd.read_excel(data_path, engine="openpyxl")
        elif data_path.suffix.lower() == ".xls":
            df = pd.read_excel(data_path, engine="xlrd")
        elif data_path.suffix.lower() == ".csv":
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        if self.config.target not in df.columns:
            raise ValueError(f"Target column '{self.config.target}' not found in data")

        self._validate_column_coverage(df)

        df = self._filter_by_year(df)
        df = self._select_features(df)
        df = self._handle_missing_values(df)
        y = df[self.config.target].copy()
        df = df.drop(columns=[self.config.target])
        df = self._encode_categoricals(df)
        df = self._encode_ordinals(df)

        y = self._preprocess_target(y)

        if self.config.preprocessing["sample_size"] is not None:
            df, y = self._sample_data(df, y)

        return df, y

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features according to configuration."""
        intended_features = set()

        feature_categories = ["numerical", "categorical", "ordinal", "dates", "binary"]

        for category in feature_categories:
            if category in self.config.features:
                intended_features.update(self.config.features[category])

        intended_features.add(self.config.target)

        columns_to_keep = [col for col in intended_features if col in df.columns]

        return df[columns_to_keep]

    def _filter_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data by minimum year if specified."""
        min_year = self.config.preprocessing["min_year"]
        if min_year is None:
            return df

        year_column = self.config.preprocessing["year_column"]
        if year_column not in df.columns:
            print(
                f"⚠️ Warning: Year column '{year_column}' not found, skipping year filtering"
            )
            return df

        initial_count = len(df)
        df_filtered = df[df[year_column] >= min_year].copy()
        filtered_count = len(df_filtered)

        print(
            f"📅 Year filtering: {initial_count} → {filtered_count} cases (kept {year_column} >= {min_year})"
        )
        return df_filtered

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values by removing affected rows."""
        missing_codes = self.config.preprocessing["missing_value_codes"]

        if not missing_codes:
            return df

        rows_to_drop = set()

        for code, columns in missing_codes.items():
            code_value = int(code) if code.lstrip("-").isdigit() else code

            for col in columns:
                if col in df.columns:
                    missing_mask = df[col] == code_value
                    print(len(df[df[col] == code_value]))
                    rows_to_drop.update(df[missing_mask].index)

        df_clean = df.drop(index=rows_to_drop)
        print(f"🗑️  Removed {len(rows_to_drop)} rows with missing values")
        return df_clean.fillna(0)

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using one-hot encoding."""
        categorical_features = self.config.features["categorical"]
        if (
            not categorical_features
            or not self.config.preprocessing["one_hot_categoricals"]
        ):
            return df

        for feature in categorical_features:
            if feature not in df.columns:
                continue

            encoder = OneHotEncoder(
                sparse_output=False,
                drop=None,
                handle_unknown="ignore",
            )

            encoded = encoder.fit_transform(df[[feature]])

            feature_names = [f"{feature}_{cat}" for cat in encoder.categories_[0]]

            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

            self.categorical_encoders[feature] = encoder

            df = pd.concat([df, encoded_df], axis=1)

            df = df.drop(columns=[feature])

        return df

    def _encode_ordinals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode ordinal features."""
        ordinal_features = self.config.features["ordinal"]

        if not ordinal_features:
            return df

        if self.config.preprocessing["one_hot_ordinals"]:

            for feature in ordinal_features:
                if feature not in df.columns:
                    continue

                encoder = OneHotEncoder(
                    sparse_output=False, drop="first", handle_unknown="ignore"
                )
                encoded = encoder.fit_transform(df[[feature]])
                feature_names = [
                    f"{feature}_{cat}" for cat in encoder.categories_[0][1:]
                ]

                encoded_df = pd.DataFrame(
                    encoded, columns=feature_names, index=df.index
                )
                self.ordinal_encoders[feature] = encoder

                df = pd.concat([df, encoded_df], axis=1)
                df = df.drop(columns=[feature])
        else:

            for feature in ordinal_features:
                if feature not in df.columns:
                    continue

                df[feature] = pd.to_numeric(df[feature], errors="coerce")

        return df

    def _preprocess_target(self, y: pd.Series) -> pd.Series:
        """Preprocess target variable with enhanced type detection."""
        if pd.api.types.is_numeric_dtype(y):
            self.target_type = "numerical"

            return y.astype(bool)
        else:
            self.target_type = "categorical"

            self.label_encoder = LabelEncoder()
            return pd.Series(self.label_encoder.fit_transform(y), index=y.index)

    def _sample_data(
        self, df: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Sample data according to configured sample size while preserving class imbalance."""
        sample_size = self.config.preprocessing["sample_size"]
        if sample_size is None:
            return df, y
        sample_size = int(sample_size)
        if sample_size >= len(df):
            return df, y
        sample_size = min(sample_size, len(df))

        np.random.seed(self.config.preprocessing.get("random_state", 42))

        # Get class distribution
        class_counts = y.value_counts()
        total_samples = len(y)

        sampled_indices = []

        for class_label, count in class_counts.items():
            # Calculate proportion of this class
            class_proportion = count / total_samples

            # Calculate target sample size for this class
            class_sample_size = int(np.round(sample_size * class_proportion))

            # Ensure we don't sample more than available
            class_sample_size = min(class_sample_size, count)

            # Get indices for this class
            class_indices = y[y == class_label].index.tolist()

            # Sample from this class
            if class_sample_size > 0:
                sampled_class_indices = np.random.choice(
                    class_indices, size=class_sample_size, replace=False
                )
                sampled_indices.extend(sampled_class_indices)

        # Convert to numpy array and shuffle to avoid any ordering bias
        sampled_indices = np.array(sampled_indices)
        np.random.shuffle(sampled_indices)

        return df.loc[sampled_indices], y.loc[sampled_indices]

    def get_dimension_info_for_afcba(self) -> dict[str, Any]:
        """Generate dimension info for AF-CBA case base creation."""
        dimension_info = {}

        feature_types = {
            "numerical": "numerical",
            "ordinal": "ordinal",
            "binary": "binary",
            "categorical": "categorical",
        }

        for feature_type, afcba_type in feature_types.items():
            features = self.config.features[feature_type]

            for feature in features:
                if feature in self.config.features["exclude"]:
                    continue

                dimension_info[feature] = {
                    "afcba_type": afcba_type,
                    "original_name": feature,
                }

        return dimension_info
