#!/usr/bin/env python3
"""Read-only byte preservation and independent CPU numerical relocation checks.

Usage (use the project's Python environment):
  python scripts/verify_restructure.py --baseline BACKUP --report /tmp/check.json
  python scripts/verify_restructure.py --baseline BACKUP --report /tmp/check.json \
      --numerical both --replay-request /tmp/replay.json

The baseline manifest is {"files": {"relative/path": {"sha256": ..., "bytes": N}}}.
New protected files are reported separately; changed/missing baseline files fail.
Numerical comparisons require EXACT full-table equality, including labels, dates,
dtypes and every value. No tolerances, fitted models, downloads or GPU operations.

Optional replay request JSON (all paths relative to the request, or absolute):
  {"daily": {"path": "daily.csv", "sha256": "...", "bytes": 123},
   "calendar": {"path": "calendar.csv", "sha256": "...", "bytes": 123},
   "forecasts": {"path": "forecasts.csv", "sha256": "...", "bytes": 123},
   "settings": {"path": "settings.json", "sha256": "...", "bytes": 123},
   "returns_verification": {"method": "verified load_returns snapshot",
                            "source_manifest_sha256": "..."}}
Daily/forecasts CSVs have a first-column date index. Calendar CSV has exactly one
date column. Settings JSON contains the complete realistic_portfolios settings
dictionary. Prepare returns locally with the project's verified snapshot loader;
this script verifies supplied bytes and the attestation format, not the financial
source provenance itself. Combine saved public forecasts locally before replay.
No observations or complete derived tables leave either worker: only SHA256s,
dimensions, coverage counts and sanitized errors enter the report.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


PROTECTED = ("outputs", "data", "configs", "supplement")
FIGURE_SCRIPT = "scripts/make_findings_figures.py"


def fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_path(root, relative):
    part = PurePosixPath(relative.replace("\\", "/"))
    if part.is_absolute() or ".." in part.parts or not part.parts or ":" in str(part):
        raise ValueError("Invalid relative manifest path")
    target = root.joinpath(*part.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Manifest path escapes root")
    return target


def protected_name(name):
    return name == FIGURE_SCRIPT or name.split("/", 1)[0] in PROTECTED


def validate_identity(entry):
    return (isinstance(entry, dict)
            and isinstance(entry.get("sha256"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", entry["sha256"]) is not None
            and type(entry.get("bytes")) is int and entry["bytes"] >= 0)


def source_identity(root):
    files = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Source symlink escapes root")
            files[path.relative_to(root).as_posix()] = fingerprint(path)
    return files


def protected_check(baseline, current, manifest, live_updates=None):
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("Baseline manifest needs a files mapping")
    selected = {}
    for raw, entry in files.items():
        name = str(PurePosixPath(raw.replace("\\", "/")))
        safe_path(baseline, name)
        safe_path(current, name)
        if name in selected:
            raise ValueError("Duplicate normalized manifest path")
        if protected_name(name):
            if not validate_identity(entry):
                raise ValueError("Protected manifest entry needs SHA256 and bytes")
            selected[name] = {"sha256": entry["sha256"].lower(), "bytes": entry["bytes"]}
    if not selected or FIGURE_SCRIPT not in selected:
        raise ValueError("Manifest must include protected files and the existing figure script")
    live_updates = live_updates or {}
    if not isinstance(live_updates, dict) or not set(live_updates).issubset(selected):
        raise ValueError("Live updates must name existing protected manifest entries")
    transitions = []
    errors = []
    verified = {"baseline": 0, "current": 0}
    for side, root in (("baseline", baseline), ("current", current)):
        for name, expected in selected.items():
            target = safe_path(root, name)
            transition = live_updates.get(name) if side == "current" else None
            if transition:
                kind = transition.get("kind")
                if (kind == "completed_stage" and name.startswith("outputs/")
                        and any(part.startswith(".revised-") for part in PurePosixPath(name).parts)):
                    completed = transition.get("completed_path", "")
                    replacement = safe_path(current, completed)
                    if (not completed.startswith("outputs/") or target.exists()
                            or not replacement.is_file() or fingerprint(replacement) != expected):
                        errors.append({"side": side, "path": name, "error": "unverified_stage_transition"})
                    else:
                        transitions.append({"path": name, "kind": kind, "completed_path": completed,
                                            "verified_identity": expected})
                        verified[side] += 1
                    continue
                if kind == "append_only_log" and name.startswith("outputs/trial-logs/") and name.endswith(".log"):
                    original = safe_path(baseline, name)
                    if target.is_file() and target.read_bytes().startswith(original.read_bytes()):
                        transitions.append({"path": name, "kind": kind,
                                            "baseline_identity": expected, "current_identity": fingerprint(target)})
                        verified[side] += 1
                    else:
                        errors.append({"side": side, "path": name, "error": "log_prefix_changed"})
                    continue
                raise ValueError("Unsupported live output transition")
            if not target.is_file():
                errors.append({"side": side, "path": name, "error": "missing"})
            elif fingerprint(target) != expected:
                errors.append({"side": side, "path": name, "error": "changed_sha256_or_size"})
            else:
                verified[side] += 1
    new = {}
    for side, root in (("baseline", baseline), ("current", current)):
        actual = set()
        for tree in PROTECTED:
            for path in (root / tree).rglob("*"):
                if path.is_file():
                    safe_path(root, path.relative_to(root).as_posix())
                    actual.add(path.relative_to(root).as_posix())
        new[side] = sorted(actual.difference(selected))
    errors.extend({"side": "baseline", "path": name, "error": "absent_from_baseline_manifest"}
                  for name in new["baseline"])
    return {"status": "passed" if not errors else "failed", "baseline_files": len(selected),
            "verified": verified, "errors": errors, "new_files": new,
            "verified_live_updates": transitions,
            "new_file_counts": {side: len(names) for side, names in new.items()},
            "new_files_policy": "Reported explicitly; additions alone do not fail."}


def normalize_request(path):
    request = read_json(path)
    verified = {"request": fingerprint(path)}
    for kind in ("daily", "calendar", "forecasts", "settings"):
        entry = request.get(kind)
        if not validate_identity(entry) or not isinstance(entry.get("path"), str):
            raise ValueError("Each replay input needs path, SHA256 and bytes")
        target = (path.parent / entry["path"]).resolve()
        actual = fingerprint(target)
        if actual != {"sha256": entry["sha256"].lower(), "bytes": entry["bytes"]}:
            raise ValueError("Replay input identity does not match")
        verified[kind] = actual
    attestation = request.get("returns_verification", {})
    if (not isinstance(attestation, dict) or not isinstance(attestation.get("method"), str)
            or not attestation["method"].strip()
            or not isinstance(attestation.get("source_manifest_sha256"), str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", attestation["source_manifest_sha256"]) is None):
        raise ValueError("Replay requires a locally verified returns provenance attestation")
    return verified


def worker_guard():
    """Enforce no file mutations, subprocesses, network connections or GPU imports."""
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    mutations = {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link", "os.symlink",
                 "os.truncate", "os.chmod", "os.chown", "os.utime", "shutil.copyfile",
                 "subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"}

    def audit(event, args):
        if event.startswith("socket.") or event in mutations:
            raise PermissionError("Read-only CPU verifier blocked an operation")
        if event == "open":
            _, mode, flags = args
            if (isinstance(mode, str) and any(letter in mode for letter in "wax+")) or (
                    isinstance(flags, int) and flags & write_flags):
                raise PermissionError("Read-only verifier blocked a file write")
        if event == "import" and args[0].split(".", 1)[0] in {"torch", "cupy", "tensorflow", "jax", "jaxlib"}:
            raise PermissionError("CPU verifier blocked a GPU-capable runtime")

    sys.addaudithook(audit)


def scalar(value, np, pd):
    if value is None or value is pd.NA:
        return ["none"]
    if value is pd.NaT:
        return ["nat"]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return [type(value).__name__, value.isoformat()]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (int, np.integer)):
        return ["int", str(value)]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return ["float", "nan" if math.isnan(number) else number.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, tuple):
        return ["tuple", [scalar(item, np, pd) for item in value]]
    raise TypeError("Unsupported table scalar type")


def table_hash(frame, np, pd):
    digest = hashlib.sha256()

    def add(item):
        digest.update(json.dumps(item, ensure_ascii=True, separators=(",", ":"),
                                 allow_nan=False).encode("utf-8"))
        digest.update(b"\n")

    add({"shape": list(frame.shape), "dtypes": [str(item) for item in frame.dtypes],
         "index_type": type(frame.index).__name__, "index_dtype": str(frame.index.dtype),
         "index_names": [scalar(item, np, pd) for item in frame.index.names],
         "column_type": type(frame.columns).__name__,
         "column_names": [scalar(item, np, pd) for item in frame.columns.names]})
    add([scalar(item, np, pd) for item in frame.columns])
    rows = (frame.itertuples(index=False, name=None) if len(frame.columns)
            else (() for _ in frame.index))
    for index, row in zip(frame.index, rows, strict=True):
        add([scalar(index, np, pd), [scalar(item, np, pd) for item in row]])
    return {"sha256": digest.hexdigest(), "rows": len(frame), "columns": len(frame.columns)}


def synthetic_fixture(np, pd):
    calendar = pd.bdate_range("2017-01-02", "2020-12-31", name="Date")
    rng = np.random.default_rng(20261022)
    t = np.arange(len(calendar), dtype=float)
    cycle = np.sin(t / 37.)
    market = .0002 + .0016 * cycle + .009 * rng.standard_normal(len(t))
    daily = pd.DataFrame({"SPY": market,
                          "TLT": -.22 * market + .005 * rng.standard_normal(len(t)),
                          "GLD": .08 * market + .006 * rng.standard_normal(len(t)),
                          "DBC": .25 * market + .007 * rng.standard_normal(len(t)),
                          "RF": .00004 + .00001 * np.cos(t / 60.)}, index=calendar)
    forecasts = pd.DataFrame(index=calendar[:-2])
    forecasts["trade_date"], forecasts["return_date"] = calendar[1:-1], calendar[2:]
    for number, model in enumerate(("arfima", "gru_univar", "garch")):
        volatility = .004 + .012 * (.5 + .5 * np.sin(t[:-2] / (29. + number * 7) + number))
        forecasts[f"{model}_bv"] = volatility ** 2
        forecasts[f"{model}_refit_asof"] = forecasts.index
    settings = {"calibration_start": "2017-01-01", "evaluation_start": "2018-01-01",
                "evaluation_end": "2020-12-31", "primary_models": ["arfima", "gru_univar"],
                "secondary_models": ["garch"], "primary_assets": ["TLT", "GLD"],
                "secondary_assets": ["DBC"], "monthly_target": .10, "monthly_multiplier": 22,
                "annual_target": .10, "stress_quantile": .9, "defensive_cap": .2,
                "costs_bps": [0, 10, 25, 100], "main_cost_bps": 10, "sharpe_hac_lags": 20,
                "monthly_hac_lags": 3, "alpha": .05,
                "windows": ["2018:2020", "2018:2018", "2019:2020"],
                "require_aligned_horizon": True}
    return daily, calendar, forecasts, settings


def replay_fixture(path, np, pd):
    identities = normalize_request(path)
    request = read_json(path)
    paths = {key: (path.parent / request[key]["path"]).resolve()
             for key in ("daily", "calendar", "forecasts", "settings")}
    daily = pd.read_csv(paths["daily"], index_col=0, parse_dates=True, float_precision="round_trip")
    forecasts = pd.read_csv(paths["forecasts"], index_col=0, parse_dates=True, float_precision="round_trip")
    raw_calendar = pd.read_csv(paths["calendar"])
    if len(raw_calendar.columns) != 1:
        raise ValueError("Calendar CSV must contain exactly one date column")
    calendar = pd.DatetimeIndex(pd.to_datetime(raw_calendar.iloc[:, 0]), name="Date")
    for frame in (daily, forecasts):
        if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
                or not frame.index.is_unique or not frame.index.is_monotonic_increasing):
            raise ValueError("Input date index must be unique, valid and increasing")
    if calendar.hasnans or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Calendar dates must be unique, valid and increasing")
    for column in forecasts:
        if column.endswith("_asof") or column in ("trade_date", "return_date", "history_first", "history_last"):
            forecasts[column] = pd.to_datetime(forecasts[column])
    settings = read_json(paths["settings"])
    if not isinstance(settings, dict):
        raise ValueError("Settings JSON needs one complete settings dictionary")
    return daily, calendar, forecasts, settings, identities


def worker(args):
    source = args.source_root.resolve()
    # -I ignores PYTHONPATH; the chosen source root is inserted before site-packages.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    sink = open(os.devnull, "w", encoding="utf-8")
    outcome = {"status": "unavailable", "tables": {}}
    phase = "install_read_only_guard"
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            worker_guard()
            phase = "import_numpy"
            np = importlib.import_module("numpy")
            phase = "import_pandas"
            pd = importlib.import_module("pandas")
            phase = "import_project_calculation_module"
            module = importlib.import_module("financial_economics." + args.module)
            # Reject inadvertent import of the other checkout or an installed copy.
            for name, loaded in tuple(sys.modules.items()):
                if name == "financial_economics" or name.startswith("financial_economics."):
                    filename = getattr(loaded, "__file__", None)
                    if filename and not Path(filename).resolve().is_relative_to(source):
                        raise RuntimeError("Calculation module loaded outside selected source root")
            tables = {}
            coverage = {}

            def evaluate(label, fixture):
                daily, calendar, forecasts, settings = fixture
                result = module.realistic_portfolios(daily, calendar, forecasts, settings)
                selected = {"performance": module._performance(result, settings),
                            "tests": module._tests(result, calendar, settings),
                            "declines": module._declines(result, settings),
                            "returns": result["returns"], "thresholds": result["thresholds"],
                            "static_controls": result["static_controls"], "decisions": result["decisions"]}
                for cost, frame in result["paths"].items():
                    selected[f"paths/{cost:g}bps"] = frame
                for cost, portfolios in result["accounting"].items():
                    for name, frame in portfolios.items():
                        selected[f"accounting/{cost:g}bps/{name}"] = frame
                for name, frame in selected.items():
                    tables[f"{label}/{name}"] = table_hash(frame, np, pd)
                coverage[label] = {"evaluation_dates": len(result["returns"]),
                                   "forecast_dates": len(forecasts), "cost_scenarios": len(result["paths"]),
                                   "portfolio_paths": sum(len(frame.columns) for frame in result["paths"].values()),
                                   "accounting_tables": sum(len(item) for item in result["accounting"].values()),
                                   "tests": len(selected["tests"]),
                                   "available_tests": int(selected["tests"].status.eq("available").sum()),
                                   "scheduled_signal_days": result["scheduled_signal_days"],
                                   "missing_signal_days": result["missing_signal_days"],
                                   "dropped_issue_dates": result["dropped_issue_dates"]}

            if args.worker_case == "synthetic":
                phase = "synthetic_fixture_and_full_table_execution"
                daily, calendar, forecasts, settings = synthetic_fixture(np, pd)
                for frequency in ("daily", "weekly", "monthly"):
                    for mode in ("paper", "annual"):
                        variant = {**settings, "rebalance_frequency": frequency, "risk_mode": mode,
                                   "initial_cash": mode == "annual", "cost_benchmark": True,
                                   "liquidate_end": mode == "annual"}
                        evaluate(f"{frequency}/{mode}", (daily, calendar, forecasts, variant))
                inputs = {name: table_hash(frame, np, pd) for name, frame in
                          (("daily", daily), ("forecasts", forecasts),
                           ("calendar", pd.DataFrame(index=calendar)))}
                inputs["settings_sha256"] = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            else:
                phase = "saved_forecast_input_loading_and_full_table_execution"
                daily, calendar, forecasts, settings, inputs = replay_fixture(args.replay_request, np, pd)
                evaluate("saved_forecasts", (daily, calendar, forecasts, settings))
                if inputs != normalize_request(args.replay_request):
                    raise RuntimeError("Replay inputs changed during numerical evaluation")
            from importlib.metadata import version
            outcome = {"status": "passed", "tables": tables, "coverage": coverage, "inputs": inputs,
                       "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "statsmodels")}}
    except Exception as error:
        # Exception messages and captured output can contain licensed observations.
        outcome = {"status": "unavailable", "error_type": type(error).__name__,
                   "phase": phase,
                   "error": "Worker could not complete the requested read-only CPU coverage.", "tables": {}}
    print(json.dumps(outcome, sort_keys=True, allow_nan=False))
    return 0 if outcome["status"] == "passed" else 2


def numerical_case(args, case):
    results = {}
    for side, source, module in (("baseline", args.baseline / "src", args.baseline_module),
                                 ("current", args.current / "src", args.current_module)):
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--_worker",
                   "--source-root", str(source), "--module", module, "--worker-case", case]
        if args.replay_request:
            command += ["--replay-request", str(args.replay_request)]
        environment = dict(os.environ)
        environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                            "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
                            "ROCR_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        try:
            completed = subprocess.run(command, cwd=args.current, env=environment,
                                       capture_output=True, text=True, timeout=args.timeout)
            result = json.loads(completed.stdout)
            if completed.returncode != 0 or result.get("status") != "passed":
                result["status"] = "unavailable"
            results[side] = result
        except (subprocess.TimeoutExpired, OSError, ValueError) as error:
            results[side] = {"status": "unavailable", "error_type": type(error).__name__,
                             "error": "Worker failed; raw output suppressed.", "tables": {}}
    if any(result["status"] != "passed" for result in results.values()):
        return {"status": "unavailable", "workers": results}
    before, after = results["baseline"], results["current"]
    names = set(before["tables"]) | set(after["tables"])
    differences = sorted(name for name in names if before["tables"].get(name) != after["tables"].get(name))
    inputs_equal = before["inputs"] == after["inputs"]
    coverage_equal = before["coverage"] == after["coverage"]
    versions_equal = before["versions"] == after["versions"]
    return {"status": "passed" if not differences and inputs_equal and coverage_equal and versions_equal else "failed",
            "comparison": "exact hashes of complete ordered table schemas, labels, dates and values",
            "compared_tables": len(names), "differing_tables": differences,
            "same_input_bytes_or_fixture": inputs_equal, "same_coverage": coverage_equal,
            "same_dependency_versions": versions_equal, "workers": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--current", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--numerical", choices=("none", "synthetic", "replay", "both"), default="synthetic")
    parser.add_argument("--replay-request", type=Path)
    parser.add_argument("--live-updates", type=Path,
                        help="Explicit JSON of byte-verified stage moves and append-only pre-existing logs")
    parser.add_argument("--baseline-module", default="paper_realistic")
    parser.add_argument("--current-module", default="studies.realistic")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--module", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", choices=("synthetic", "replay"), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return worker(args)
    if args.baseline is None or args.report is None:
        parser.error("--baseline and --report are required")
    args.baseline, args.current, args.report = (path.resolve() for path in (args.baseline, args.current, args.report))
    if args.baseline == args.current:
        parser.error("Baseline and current roots must be distinct")
    if not (args.baseline / "src").is_dir() or not (args.current / "src").is_dir():
        parser.error("Both roots must contain src directories")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    denied = [args.baseline, args.current / "src", args.current / "scripts", args.current / ".git"]
    denied += [args.current / tree for tree in PROTECTED]
    if any(args.report.is_relative_to(root.resolve()) for root in denied):
        parser.error("Report must be outside baseline, source, scripts and protected trees")
    if args.replay_request:
        args.replay_request = args.replay_request.resolve()
    if args.numerical in ("replay", "both") and args.replay_request is None:
        parser.error("Replay requires --replay-request")
    if args.replay_request:
        if args.report == args.replay_request:
            parser.error("Report must not overwrite the replay request")
        try:
            request_inputs = read_json(args.replay_request)
            if any(args.report == (args.replay_request.parent / entry["path"]).resolve()
                   for entry in request_inputs.values()
                   if isinstance(entry, dict) and isinstance(entry.get("path"), str)):
                parser.error("Report must not overwrite a replay input")
        except (OSError, ValueError, AttributeError):
            parser.error("Replay request could not be read as a JSON object")
    report = {"status": "failed", "limitations": [
        "Synthetic coverage tests portfolio execution and inference; it does not refit forecasting models.",
        "Exact comparison requires the same Python executable and dependency versions for both workers.",
        "Saved-forecast replay runs only when explicitly requested with preverified local inputs.",
        "Replay verifies caller-supplied return bytes and provenance attestation, not the source loader itself.",
        "Bootstrap intervals, forecast scoring/refits, GPU behavior and provenance compatibility are separate coverage."]}
    try:
        manifest_path = args.baseline / "baseline-manifest.json"
        manifest_identity = fingerprint(manifest_path)
        manifest = read_json(manifest_path)
        report["baseline_manifest"] = manifest_identity
        live_updates = read_json(args.live_updates) if args.live_updates else {}
        live_identity = fingerprint(args.live_updates) if args.live_updates else None
        report["live_updates_identity"] = live_identity
        report["protected"] = protected_check(args.baseline, args.current, manifest, live_updates)
        source_before = {"baseline": source_identity(args.baseline / "src"),
                         "current": source_identity(args.current / "src")}
        report["source_before"] = source_before
        if args.replay_request:
            report["replay_input_identity"] = normalize_request(args.replay_request)
        report["numerical"] = {}
        cases = {"none": (), "synthetic": ("synthetic",), "replay": ("replay",),
                 "both": ("synthetic", "replay")}[args.numerical]
        if report["protected"]["status"] == "passed":
            for case in cases:
                report["numerical"][case] = numerical_case(args, case)
        elif cases:
            report["numerical"]["coverage"] = {"status": "unavailable", "error": "Protected bytes failed verification."}
        if "replay" not in cases:
            report["saved_forecast_replay"] = "unavailable: not requested"
        if not cases:
            report["numerical_coverage"] = "unavailable: explicitly disabled"
        source_after = {"baseline": source_identity(args.baseline / "src"),
                        "current": source_identity(args.current / "src")}
        report["source_after"] = source_after
        report["source_bytes_unchanged"] = source_before == source_after
        report["baseline_manifest_unchanged"] = fingerprint(manifest_path) == manifest_identity
        report["protected_after"] = protected_check(args.baseline, args.current, manifest, live_updates)
        if args.live_updates and fingerprint(args.live_updates) != live_identity:
            raise ValueError("Live output declaration changed during verification")
        inputs_unchanged = (not args.replay_request or
                            normalize_request(args.replay_request) == report["replay_input_identity"])
        report["input_bytes_unchanged"] = inputs_unchanged
        passed = (report["protected"]["status"] == report["protected_after"]["status"] == "passed"
                  and report["source_bytes_unchanged"] and report["baseline_manifest_unchanged"]
                  and inputs_unchanged and all(item["status"] == "passed" for item in report["numerical"].values()))
        report["status"] = "passed" if passed else "failed"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = "Verification could not complete. Raw exception text suppressed to protect input observations."
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report),
                      "protected_files": report.get("protected", {}).get("baseline_files", 0),
                      "numerical": {name: item["status"] for name, item in report.get("numerical", {}).items()}}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
