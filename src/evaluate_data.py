import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from case_base import (
    create_afcba_case_base_from_preprocessed_data,
)
from experiment import AdmissionProcessor


def analyse_all_datasets(processors) -> dict[str, Any]:
    """Analyse CB consistency and class balance for all available processors."""
    results = {}

    print(f"📊 Analysing {len(processors)} dataset(s)...")

    for processor in processors:
        dataset_name = processor.get_dataset_name()
        print(f"\n🔍 Analysing {dataset_name}...")

        try:
            if not processor.config_path.exists() or not processor.data_path.exists():
                print(f"⚠️  Skipping {dataset_name}: missing files")
                continue

            X, _, y, _ = processor.prepare_data(None, 42)
            print(f"✅ Preprocessed: {len(X):,} samples, {len(X.columns)} features")

            case_base = create_afcba_case_base_from_preprocessed_data(
                X,
                y,
                min_corr=0.0,
                inference_method="logreg",
                use_harmonic_authoritativeness=False,
            )

            class_counts = Counter(y)
            class_balance_ratio = min(class_counts.values()) / max(
                class_counts.values()
            )

            # Use the new enhanced method to get detailed inconsistency information
            detailed_report = case_base.get_detailed_inconsistency_report()
            consistency_percentage = detailed_report["consistency_analysis"][
                "consistency_percentage"
            ]
            n_remove = detailed_report["consistency_analysis"]["inconsistent_cases"]

            results[dataset_name] = {
                "total_cases": len(case_base),
                "feature_count": len(case_base.feature_columns),
                "class_distribution": dict(class_counts),
                "class_balance_ratio": round(class_balance_ratio, 4),
                "consistency_percentage": round(consistency_percentage, 2),
                "n_remove": n_remove,
                "inconsistency_details": detailed_report,  # Include full inconsistency analysis
            }

            # Enhanced reporting with specific cases to remove
            if n_remove > 0:
                print(
                    f"📊 {dataset_name}: {len(case_base):,} cases, {consistency_percentage:.2f}% consistent"
                )
                print(
                    f"   🗑️  Must remove {n_remove} cases ({100-consistency_percentage:.2f}%) for consistency"
                )
                print(
                    f"   📋 Cases to remove (by index): {detailed_report['cases_to_remove']['indices']}"
                )
                print(f"   ⚖️  Class balance ratio: {class_balance_ratio:.3f}")

                # Show details of first few cases to be removed (for brevity)
                cases_to_show = min(
                    5, len(detailed_report["cases_to_remove"]["detailed_cases"])
                )
                if cases_to_show > 0:
                    print(f"   🔍 First {cases_to_show} cases to remove:")
                    for i, case in enumerate(
                        detailed_report["cases_to_remove"]["detailed_cases"][
                            :cases_to_show
                        ]
                    ):
                        orig_idx = case["original_dataframe_index"]
                        case_idx = case["original_index"]
                        outcome = case.get(case_base.target_column, "unknown")
                        print(
                            f"      • Row {orig_idx} (case {case_idx}): outcome={outcome}"
                        )

                    if n_remove > cases_to_show:
                        print(f"      ... and {n_remove - cases_to_show} more cases")
            else:
                print(
                    f"📊 {dataset_name}: {len(case_base):,} cases, {consistency_percentage:.2f}% consistent (no removals needed)"
                )
                print(f"   ⚖️  Class balance ratio: {class_balance_ratio:.3f}")

        except Exception as e:
            print(f"❌ Failed to analyse {dataset_name}: {e}")
            continue

    return results


def export_results_to_json(results: dict[str, Any]) -> str:
    """Export results to JSON file in output directory."""
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    export_data = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_datasets": len(results),
        "dataset_results": results,
        "summary": {
            "average_consistency": (
                float(
                    round(
                        sum(
                            float(r["consistency_percentage"]) for r in results.values()
                        )
                        / len(results),
                        2,
                    )
                )
                if results
                else 0.0
            ),
            "datasets_by_consistency": sorted(
                [
                    (name, float(data["consistency_percentage"]))
                    for name, data in results.items()
                ],
                key=lambda x: x[1],
                reverse=True,
            ),
            "total_cases_to_remove": int(
                sum(int(r.get("n_remove", 0)) for r in results.values())
            ),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset_analysis_detailed_{timestamp}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"\n💾 Detailed results exported to: {output_path}")
    return str(output_path)


def export_removal_lists_to_csv(results: dict[str, Any]) -> None:
    """Export lists of cases to remove for each dataset as separate CSV files."""
    import pandas as pd

    output_dir = Path("output") / "removal_lists"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_name, data in results.items():
        if data.get("n_remove", 0) > 0:
            inconsistency_details = data.get("inconsistency_details", {})
            cases_to_remove = inconsistency_details.get("cases_to_remove", {}).get(
                "detailed_cases", []
            )

            if cases_to_remove:
                # Create DataFrame from the cases to remove
                removal_df = pd.DataFrame(cases_to_remove)

                # Save as CSV
                csv_filename = f"{dataset_name.lower()}_cases_to_remove_{timestamp}.csv"
                csv_path = output_dir / csv_filename
                removal_df.to_csv(csv_path, index=False)

                print(f"📄 {dataset_name}: Cases to remove exported to {csv_path}")


def dataset_consistency_and_balance(processors):
    """Single test function for CB consistency and class balance analysis with detailed removal reporting."""
    print("🚀 Starting detailed dataset analysis...")

    results = analyse_all_datasets(processors)

    assert len(results) > 0, "No datasets were successfully analysed"

    # json_path = export_results_to_json(results)
    export_removal_lists_to_csv(results)

    print("\n" + "=" * 80)
    print("📋 DETAILED SUMMARY:")
    print("=" * 80)

    total_cases = sum(r["total_cases"] for r in results.values())
    total_removals = sum(r.get("n_remove", 0) for r in results.values())

    for dataset_name, data in results.items():
        n_remove = data.get("n_remove", 0)
        removal_pct = (
            (n_remove / data["total_cases"]) * 100 if data["total_cases"] > 0 else 0
        )

        print(f"\n🗂️  {dataset_name}:")
        print(f"   • Total cases: {data['total_cases']:,}")
        print(f"   • Consistency: {data['consistency_percentage']:.2f}%")
        print(f"   • Cases to remove: {n_remove} ({removal_pct:.1f}%)")
        print(f"   • Class balance ratio: {data['class_balance_ratio']:.3f}")
        indices = []
        if n_remove > 0:
            indices = (
                data.get("inconsistency_details", {})
                .get("cases_to_remove", {})
                .get("indices", [])
            )
            if len(indices) <= 10:
                print(f"   • Indices to remove: {indices}")
            else:
                print(
                    f"   • Indices to remove: {indices[:5]} ... {indices[-3:]} (showing first 5 and last 3)"
                )

    avg_consistency = sum(r["consistency_percentage"] for r in results.values()) / len(
        results
    )
    overall_removal_pct = (total_removals / total_cases) * 100 if total_cases > 0 else 0

    print("\n🎯 Overall Statistics:")
    print(f"   • Average consistency: {avg_consistency:.2f}%")
    print(f"   • Total cases across all datasets: {total_cases:,}")
    print(
        f"   • Total cases requiring removal: {total_removals:,} ({overall_removal_pct:.1f}%)"
    )
    print(
        f"   • Resulting dataset sizes after cleanup: {total_cases - total_removals:,}"
    )

    print("\n🎉 Detailed analysis complete! Results saved to:")
    # print(f"   📊 JSON report: {Path(json_path).name}")
    print("   📄 CSV removal lists: output/removal_lists/")

    return indices


if __name__ == "__main__":
    processors = [
        AdmissionProcessor(),
        # ChurnProcessor(),
        # GTDProcessor()  # Uncomment if you want to include this dataset
    ]
    dataset_consistency_and_balance(processors)
