"""Keep manual acceptance evidence separate from the automated physical traffic gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument(
        "--scope", type=Path, help="Predeclared MODE:CASE → reason for inapplicable cases"
    )
    record = sub.add_parser("record")
    record.add_argument("--ledger", type=Path, required=True)
    record.add_argument("--mode", choices=["queue", "vllm-sleep"], required=True)
    record.add_argument("--case", required=True)
    record.add_argument(
        "--status", choices=["PASS", "FAIL", "BLOCKED", "NOT_APPLICABLE"], required=True
    )
    record.add_argument("--evidence", type=Path, required=True)
    record.add_argument("--notes", required=True)
    report = sub.add_parser("report")
    report.add_argument("--ledger", type=Path, required=True)
    report.add_argument(
        "--traffic", type=Path, nargs=2, required=True, help="One queue and one sleep report.json"
    )
    args = parser.parse_args()
    plan_path = ROOT / "docs/release/native-acceptance.json"
    if args.command == "init":
        if args.output.exists():
            raise SystemExit("Refusing to overwrite existing acceptance evidence")
        scope = json.loads(args.scope.read_text()) if args.scope else {}
        plan = json.loads(plan_path.read_text())
        cases = {
            f"{mode}:{case['id']}": {"status": "UNRUN", "history": []}
            for case in plan["cases"]
            for mode in case["modes"]
        }
        if set(scope) - cases.keys() or any(not str(reason).strip() for reason in scope.values()):
            raise SystemExit(
                "Scope exclusions must name existing mode/case IDs and explain applicability"
            )
        save(args.output, {"plan_sha256": digest(plan_path), "scope": scope, "cases": cases})
        print(f"Created {len(cases)} unexecuted mode/case entries")
        return
    ledger = json.loads(args.ledger.read_text())
    if ledger["plan_sha256"] != digest(plan_path):
        raise SystemExit("Acceptance plan changed; review and initialize evidence for the new plan")
    if args.command == "record":
        key = f"{args.mode}:{args.case}"
        if key not in ledger["cases"]:
            raise SystemExit("Unknown mode/case")
        if args.status == "NOT_APPLICABLE" and key not in ledger["scope"]:
            raise SystemExit("NOT_APPLICABLE requires a predeclared scope exclusion")
        if not args.notes.strip():
            raise SystemExit("Describe the observed outcome")
        entry = {
            "status": args.status,
            "at": datetime.now(UTC).isoformat(),
            "notes": args.notes,
            "evidence": str(args.evidence.resolve(strict=True)),
            "sha256": digest(args.evidence),
        }
        ledger["cases"][key]["history"].append(entry)
        ledger["cases"][key].update({k: v for k, v in entry.items() if k != "history"})
        save(args.ledger, ledger)
        return
    errors = []
    reports = [json.loads(path.read_text()) for path in args.traffic]
    if {r["mode"] for r in reports} != {"queue", "vllm-sleep"}:
        errors.append("Require one physical traffic report per native mode")
    if any(not r.get("traffic_gate_passed") for r in reports):
        errors.append("A physical traffic gate has not passed")
    if reports[0].get("build_sha256") != reports[1].get("build_sha256"):
        errors.append("Traffic reports identify different builds")
    for report_path in args.traffic:
        hashes = json.loads((report_path.parent / "evidence-hashes.json").read_text())
        for name, expected in hashes.items():
            path = report_path.parent / name
            if not path.is_file() or digest(path) != expected:
                errors.append(f"Traffic evidence changed or missing: {path}")
    for key, entry in ledger["cases"].items():
        if entry["status"] not in ("PASS", "NOT_APPLICABLE"):
            errors.append(f"{key}: {entry['status']}")
        elif (
            not Path(entry["evidence"]).is_file()
            or digest(Path(entry["evidence"])) != entry["sha256"]
        ):
            errors.append(f"{key}: evidence changed or missing")
    print(
        json.dumps(
            {
                "counts": dict(Counter(e["status"] for e in ledger["cases"].values())),
                "evidence_complete": not errors,
                "unresolved": errors,
            },
            indent=2,
        )
    )
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
