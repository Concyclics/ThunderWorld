"""Summarize completed development costs and paired errors without final-set claims."""
import argparse
import collections
import gzip
import json
from pathlib import Path

from baselines import sha, write


def interval(values):
    return {"min": min(values), "max": max(values)} if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.study / "balanced_followup/development"
    manifest = json.loads((root / "manifest.json").read_text())
    quality = json.loads((root / "report/quality.json").read_text())
    cohort = json.loads((args.study / "cohort.json").read_text())
    if not quality["complete"] or quality["manifest_sha256"] != sha(root / "manifest.json"):
        raise ValueError("Complete matching development evidence required")
    allowed = {entry["case"]["id"] for entry in cohort["cases"]["development"]}
    if {entry["case"]["id"] for entry in manifest["cases"]} != allowed:
        raise ValueError("Development cohort differs from frozen inputs")
    catalog = {}
    for sub in ("candidates", "ffn_followup/candidates", "ffn_followup_k/candidates", "balanced_followup/candidates"):
        for candidate in json.loads((args.study / sub / "candidates.json").read_text())["candidates"]:
            catalog[candidate["name"]] = candidate
    costs, receipts = {}, []
    for name, method in manifest["methods"].items():
        asset_files_bytes = None
        if method.get("assets"):
            bank = Path(method["assets"])
            if sha(bank / "manifest.json") != method["assets_sha256"]:
                raise ValueError("Changed deployed asset manifest")
            assets = json.loads((bank / "manifest.json").read_text())
            asset_files_bytes = sum((bank / entry["file"]).stat().st_size for entry in assets["modules"].values())
        ratios, allocated, reserved, dtype = [], [], [], collections.defaultdict(list)
        work, traffic, terms = (collections.defaultdict(list) for _ in range(3))
        union_bytes = set()
        for case in manifest["cases"]:
            folder = root / "generation" / name / case["case"]["id"]
            path = folder / "result.json"
            result = json.loads(path.read_text())
            if not result["complete"] or result["identity"]["manifest_sha256"] != quality["manifest_sha256"]:
                raise ValueError("Mismatched request evidence")
            receipts.append({"method": name, "case": case["case"]["id"], "path": str(path), "sha256": sha(path)})
            for row in result["rows"]:
                if row["performance_eligible"] or row["debug"] != "off" or row["steps"] != 35 or row["calls"] != 70:
                    raise ValueError("Unexpected execution settings")
                if row.get("actual_mma_ratio") is not None:
                    if row["actual_mma_ratio"] > method["config"]["compute_budget"]:
                        raise ValueError("Request exceeded its hard budget")
                    ratios.append(row["actual_mma_ratio"])
                allocated.append(row["peak_allocated_bytes"])
                reserved.append(row["peak_reserved_bytes"])
            ledger_path = folder / "compute.json"
            if ledger_path.exists():
                if any(row.get("compute_sha256") != sha(ledger_path) for row in result["rows"]):
                    raise ValueError("Changed request ledger")
                ledger = json.loads(ledger_path.read_text())
                union_bytes.add(ledger["w1_support_union_bytes"])
                for key, value in ledger["mma_by_dtype"].items():
                    dtype[key].append(value)
                for dest, key in ((work, "non_mma_work"), (traffic, "packing_traffic")):
                    for label, value in ledger[key].items():
                        dest[label].append(value)
                current = collections.Counter()
                for row in ledger["rows"]:
                    for kind, items in row["mma_by_dtype"].items():
                        for term, value in items.items():
                            current[f'{row["kind"]}.{kind}.{term}'] += value * row["calls"]
                for term, value in current.items():
                    terms[term].append(value / ledger["base_mma"])
        if len(union_bytes) > 1:
            raise ValueError("Weight support union changed by request")
        costs[name] = {
            "actual_mma_ratio": interval(ratios),
            "peak_allocated_bytes": interval(allocated),
            "peak_reserved_bytes": interval(reserved),
            "mma_by_dtype": {k: interval(v) for k, v in dtype.items()},
            "mma_terms_over_C0": {k: interval(v) for k, v in terms.items()},
            "non_mma_work": {k: interval(v) for k, v in work.items()},
            "packing_traffic": {k: interval(v) for k, v in traffic.items()},
            "w1_support_union_bytes": next(iter(union_bytes), None),
            "offline_tensor_file_bytes": asset_files_bytes,
            "offline_asset_cost": catalog.get(name, {}).get("cost"),
        }
    fields = ("lpips", "ssim_mean", "relative_l2", "psnr_db")
    values = collections.defaultdict(dict)
    for record in quality["seed_metrics"]:
        for row in record["rows"]:
            values[record["benchmark"], record["name"]][record["case_id"], row["seed"]] = row["metrics"]["video"]
    paired = {}
    for (benchmark, name), candidate in values.items():
        for reference in ("fp8", "b2.5_marginal"):
            if name == reference:
                continue
            base = values[benchmark, reference]
            if candidate.keys() != base.keys():
                raise ValueError("Incomplete metric pairs")
            result = {}
            for field in fields:
                differences = [(key, value[field] - base[key][field]) for key, value in candidate.items()]
                worst_key, worst_difference = (min if field in ("ssim_mean", "psnr_db") else max)(differences, key=lambda row: row[1])
                result[field] = {
                    "mean_candidate_minus_reference": sum(v for _, v in differences) / len(differences),
                    "worst_seed_difference": worst_difference,
                    "worst_seed": {"case": worst_key[0], "seed": worst_key[1]},
                    "degraded_seed_count": sum(v < 0 if field in ("ssim_mean", "psnr_db") else v > 0 for _, v in differences),
                    "seed_count": len(differences),
                }
            paired[f"{benchmark}|{name}|{reference}"] = result
    payload = {
        "complete": True, "manifest_sha256": quality["manifest_sha256"],
        "cohort_sha256": sha(args.study / "cohort.json"),
        "cohort_counts": {k: len(v) for k, v in cohort["cases"].items()},
        "generated_evaluation_videos": len(manifest["methods"]) * sum(len(case["seeds"]) for case in manifest["cases"]),
        "summaries": quality["summaries"], "paired_development_differences": paired,
        "costs": costs, "request_receipts": receipts,
        "performance_eligible": False, "final_quality_qualified": False,
        "scope": "Exploratory new development scenes only; paired differences are descriptive, without confidence intervals. No validation/final seed was run. MMA, modeled packing work and measured allocator peaks are distinct; no shared-GPU speed claim.",
    }
    write(args.output, payload)
    ffn_manifest = json.loads((args.study / "ffn_calibration/manifest.json").read_text())
    records = []
    for case, digest in ffn_manifest["cases"].items():
        path = args.study / "ffn_calibration" / (case + ".json")
        if sha(path) != digest:
            raise ValueError("Changed FFN calibration evidence")
        records.append(json.loads(path.read_text()))
    (args.output.parent / "ffn_calibration_interventions.json.gz").write_bytes(gzip.compress(json.dumps({"manifest": ffn_manifest, "cases": records}, allow_nan=False).encode(), mtime=0))
    print(json.dumps({"complete": True, "output": str(args.output), "videos": payload["generated_evaluation_videos"]}))


if __name__ == "__main__":
    main()
