import os

import numpy as np
import pandas as pd


# url: https://github.com/propublica/compas-analysis
def load_COMPAS():
    raw_path = "data/compas-scores.csv"
    prepared_path = "data/compas.csv"
    # Ignored feature(s):
    #     id,
    #     name,
    #     first,
    #     last,
    #     compas_screening_date,
    #     dob,
    #     age_cat,
    #     num_r_cases,
    #     decile_score,
    #     days_b_screening_arrest,
    #     c_case_number,
    #     c_offense_date,
    #     c_arrest_date,
    #     c_days_from_compas,
    #     c_charge_desc,
    #     r_case_number,
    #     r_charge_degree,
    #     r_days_from_arrest,
    #     r_offense_date,
    #     r_charge_desc,
    #     r_jail_in,
    #     r_jail_out,
    #     is_violent_recid,
    #     num_vr_cases,
    #     vr_case_number,
    #     vr_charge_degree,
    #     vr_offense_date,
    #     vr_charge_desc,
    #     v_type_of_assessment,
    #     v_decile_score,
    #     v_score_text,
    #     v_screening_date,
    #     type_of_assessment,
    #     decile_score,
    #     score_text,
    #     screening_date,
    feature_set = [
        "sex",
        "age",
        "race",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_jail_in",
        "c_jail_out",
        "c_charge_degree",
        "is_recid",
    ]
    if not os.path.exists(prepared_path):
        df = pd.read_csv(raw_path)
        df = df[feature_set]
        df["c_jail_in"] = pd.to_datetime(df["c_jail_in"])
        df.dropna(subset=["c_jail_in"], inplace=True)
        df["c_jail_out"] = pd.to_datetime(df["c_jail_out"])
        df.dropna(subset=["c_jail_out"], inplace=True)
        df["jailtime"] = (
            ((df["c_jail_out"] - df["c_jail_in"]) / np.timedelta64(1, "D"))
            .round(0)
            .astype(int)
        )
        df.drop(columns={"c_jail_in", "c_jail_out"}, inplace=True)
        df.dropna(subset=["is_recid"], inplace=True)
        df.to_csv(prepared_path, index=False)
    else:
        df = pd.read_csv(prepared_path)
    return df


load_COMPAS()
