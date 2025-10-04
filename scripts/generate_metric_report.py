"""
Metric Standardization and Auto-Report Generation

Collects metrics from E2E test runs and generates:
- JSON/CSV metric cards
- Auto-updated README tables
- CI artifact summaries

Usage:
    python scripts/generate_metric_report.py --input results/e2e --output results/metric_summary.md
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import pandas as pd


class MetricReportGenerator:
    """Generate standardized metric reports from E2E results."""

    METRIC_SCHEMA = {
        "final_residual": {"description": "Final PDE residual", "format": ":.2e"},
        "max_vorticity": {"description": "Maximum vorticity magnitude", "format": ":.4f"},
        "conservation_violation": {"description": "Mass/energy conservation error", "format": ":.2e"},
        "lambda_estimate": {"description": "Estimated lambda parameter", "format": ":.6f"},
        "seed_sensitivity": {"description": "Std dev across seeds", "format": ":.2e"},
        "benchmark_time": {"description": "Total execution time (s)", "format": ":.1f"}
    }

    def __init__(self, input_dir: Path, output_path: Path):
        self.input_dir = input_dir
        self.output_path = output_path
        self.metrics_data = []

    def collect_metrics(self):
        """Collect all metric cards from input directory."""
        print(f"[*] Collecting metrics from: {self.input_dir}")

        metric_files = list(self.input_dir.rglob("metric_card.json"))
        if not metric_files:
            print(f"[!] No metric cards found in {self.input_dir}")
            return

        for metric_file in metric_files:
            with open(metric_file) as f:
                data = json.load(f)
                self.metrics_data.append({
                    "source": str(metric_file.parent.name),
                    **data.get("results", {}).get("final_metrics", {})
                })

        print(f"[+] Collected {len(self.metrics_data)} metric cards")

    def generate_csv(self) -> Path:
        """Generate CSV metric export."""
        if not self.metrics_data:
            return None

        df = pd.DataFrame(self.metrics_data)
        csv_path = self.output_path.parent / "metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[+] CSV exported: {csv_path}")
        return csv_path

    def generate_markdown_table(self) -> str:
        """Generate markdown table for README."""
        if not self.metrics_data:
            return "No metrics available."

        # Table header
        headers = ["Test Case"] + [
            self.METRIC_SCHEMA[k]["description"]
            for k in self.METRIC_SCHEMA.keys()
        ]
        md = "| " + " | ".join(headers) + " |\n"
        md += "|" + "|".join([" --- "] * len(headers)) + "|\n"

        # Table rows
        for metric in self.metrics_data:
            row = [metric.get("source", "Unknown")]
            for key, schema in self.METRIC_SCHEMA.items():
                value = metric.get(key)
                if value is not None:
                    if isinstance(value, (int, float)):
                        row.append(f"{value:{schema['format']}}")
                    else:
                        row.append(str(value))
                else:
                    row.append("N/A")
            md += "| " + " | ".join(row) + " |\n"

        return md

    def generate_summary_report(self):
        """Generate comprehensive summary markdown."""
        print("[*] Generating summary report...")

        report = f"""# E2E Metric Summary Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Test Cases**: {len(self.metrics_data)}

---

## Metric Overview

{self.generate_markdown_table()}

---

## Metric Definitions

"""
        for key, schema in self.METRIC_SCHEMA.items():
            report += f"- **{schema['description']}**: `{key}`\n"

        report += """
---

## How to Update

This report is auto-generated from E2E test results:

```bash
# Run E2E tests
pytest tests_e2e/ --json-report --json-report-file=results/e2e/metrics.json

# Generate report
python scripts/generate_metric_report.py \\
    --input results/e2e \\
    --output results/metric_summary.md
```

To add this table to README.md:

1. Copy the markdown table from above
2. Paste into README.md under "## E2E Validation Results"
3. Commit changes

---

## CI Integration

This report is automatically updated by GitHub Actions on every push to `main`.

See: `.github/workflows/e2e-metrics.yml`
"""

        with open(self.output_path, "w") as f:
            f.write(report)

        print(f"[+] Summary report saved: {self.output_path}")

    def generate_ci_artifact(self) -> Dict[str, Any]:
        """Generate CI artifact JSON for GitHub Actions."""
        artifact = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(self.metrics_data),
            "metrics": self.metrics_data,
            "summary": {
                "avg_residual": self._calculate_average("final_residual"),
                "avg_lambda": self._calculate_average("lambda_estimate"),
                "total_runtime": self._calculate_sum("benchmark_time")
            }
        }

        artifact_path = self.output_path.parent / "ci_metrics.json"
        with open(artifact_path, "w") as f:
            json.dump(artifact, f, indent=2)

        print(f"[+] CI artifact saved: {artifact_path}")
        return artifact

    def _calculate_average(self, key: str) -> float:
        """Calculate average of a metric."""
        values = [m[key] for m in self.metrics_data if key in m and m[key] is not None]
        return sum(values) / len(values) if values else 0.0

    def _calculate_sum(self, key: str) -> float:
        """Calculate sum of a metric."""
        values = [m[key] for m in self.metrics_data if key in m and m[key] is not None]
        return sum(values) if values else 0.0

    def run(self):
        """Execute full report generation pipeline."""
        self.collect_metrics()
        if self.metrics_data:
            self.generate_csv()
            self.generate_summary_report()
            self.generate_ci_artifact()
        else:
            print("[!] No metrics to process")


def main():
    parser = argparse.ArgumentParser(description="Generate E2E Metric Reports")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/e2e"),
        help="Input directory with metric cards"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metric_summary.md"),
        help="Output markdown report path"
    )
    args = parser.parse_args()

    generator = MetricReportGenerator(args.input, args.output)
    generator.run()


if __name__ == "__main__":
    main()
