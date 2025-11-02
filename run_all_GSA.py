#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_GSA.py — One-click pipeline for AquaCrop GSA (Morris → eFAST → Daily eFAST)
- Steps:
    1) Year-type classification (Q25/Q75, with ET0/Tx/Tn metrics)
    2) Morris screening (51 params)
    3) eFAST on Yield (per typical years)
    4) Time-resolved eFAST on Canopy Cover (daily)
    5) Time-resolved eFAST on Biomass (daily)
- Features:
    * Logging to results/logs
    * Idempotent & resumable (markers), use --force to re-run
    * Per-step on/off toggles
Usage:
    python run_all_GSA.py
    python run_all_GSA.py --no-year-type --only morris
    python run_all_GSA.py --force
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------- Paths (edit if needed) ----------
ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "year_type": ROOT / "year_type_classification.py",
    "morris": ROOT / "Morris_Yield.py",
    "efast_yield": ROOT / "eFAST_Yield.py",
    "efast_cc_daily": ROOT / "eFAST_CC_daily.py",
    "efast_biomass_daily": ROOT / "eFAST_Biomass_daily.py",
}

RESULTS_DIR = ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
MARK_DIR = RESULTS_DIR / "markers"

# ---------- Utilities ----------
def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def ensure_dirs():
    for d in [RESULTS_DIR, LOG_DIR, MARK_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def marker_path(step: str) -> Path:
    return MARK_DIR / f"{step}.done"

def run_step(step_key: str, description: str, force: bool = False) -> bool:
    """
    Run one step script via subprocess, with logging and marker-based resume.
    Returns True if successful or already done; False if failed.
    """
    step_script = SCRIPTS[step_key]
    mk = marker_path(step_key)

    if not step_script.exists():
        log(f"✖ Script not found for step '{step_key}': {step_script}")
        return False

    if mk.exists() and not force:
        log(f"✓ Skip (already done): {description} [{step_script.name}]")
        return True

    # Prepare log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_log = LOG_DIR / f"{timestamp}_{step_key}.out.log"
    stderr_log = LOG_DIR / f"{timestamp}_{step_key}.err.log"

    log(f"▶ Start: {description} [{step_script.name}]")
    t0 = time.time()
    with open(stdout_log, "wb") as out, open(stderr_log, "wb") as err:
        try:
            proc = subprocess.run(
                [sys.executable, str(step_script)],
                cwd=str(ROOT),
                stdout=out,
                stderr=err,
                check=False,
            )
        except Exception as e:
            log(f"✖ Exception while running {step_script.name}: {e}")
            return False

    dt = time.time() - t0

    if proc.returncode == 0:
        mk.touch()
        log(f"✓ Done in {dt:.1f}s: {description} → logs: {stdout_log.name}, {stderr_log.name}")
        return True
    else:
        log(f"✖ Failed (code={proc.returncode}) for {description}. See logs: {stdout_log.name}, {stderr_log.name}")
        return False

def parse_args():
    p = argparse.ArgumentParser(description="One-click runner for AquaCrop GSA pipeline")
    p.add_argument("--force", action="store_true", help="Re-run steps even if markers exist")
    p.add_argument("--no-year-type", action="store_true", help="Skip year-type classification step")
    p.add_argument("--no-morris", action="store_true", help="Skip Morris step")
    p.add_argument("--no-efast-yield", action="store_true", help="Skip eFAST (Yield) step")
    p.add_argument("--no-efast-cc", action="store_true", help="Skip time-resolved eFAST (Canopy Cover) step")
    p.add_argument("--no-efast-biomass", action="store_true", help="Skip time-resolved eFAST (Biomass) step")
    p.add_argument("--only", choices=["year_type", "morris", "efast_yield", "efast_cc_daily", "efast_biomass_daily"],
                   help="Run a single step only")
    return p.parse_args()

def main():
    ensure_dirs()
    args = parse_args()

    steps = [
        ("year_type", "Year-type classification (Q25/Q75 with ET0/Tx/Tn)"),
        ("morris", "Morris screening (51 parameters)"),
        ("efast_yield", "eFAST on yield (per representative years)"),
        ("efast_cc_daily", "Time-resolved eFAST (Canopy Cover, daily)"),
        ("efast_biomass_daily", "Time-resolved eFAST (Biomass, daily)"),
    ]

    # Filter by flags
    if args.only:
        steps = [s for s in steps if s[0] == args.only]
    else:
        filtered = []
        for key, desc in steps:
            if key == "year_type" and args.no_year_type:
                continue
            if key == "morris" and args.no_morris:
                continue
            if key == "efast_yield" and args.no_efast_yield:
                continue
            if key == "efast_cc_daily" and args.no_efast_cc:
                continue
            if key == "efast_biomass_daily" and args.no_efast_biomass:
                continue
            filtered.append((key, desc))
        steps = filtered

    log("========== AquaCrop GSA Pipeline ==========")
    log(f"Root: {ROOT}")
    log(f"Python: {sys.executable}")
    log("-------------------------------------------")
    ok_all = True
    for key, desc in steps:
        ok = run_step(key, desc, force=args.force)
        ok_all = ok_all and ok

    log("-------------------------------------------")
    if ok_all:
        log("🎉 All requested steps completed successfully.")
        sys.exit(0)
    else:
        log("⚠ Some steps failed. Please check logs under results/logs/")
        sys.exit(1)

if __name__ == "__main__":
    main()
