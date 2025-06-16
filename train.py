#!/usr/bin/env python3
"""Training script replacing train.bash.

This script iterates over jobsets and invokes ``cqsim.py`` for each one.
It resumes from the last saved weights and periodically runs validation.
"""

import argparse
import subprocess
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
CFG_FILE = "config_sys.set"  # relative to SRC_DIR
JOBSET_DIR = PROJECT_ROOT / "data" / "jobsets"
RESULT_DIR = PROJECT_ROOT / "data" / "results" / "theta"
DEBUG_DIR = PROJECT_ROOT / "data" / "debug" / "theta"
WEIGHT_BASENAME = PROJECT_ROOT / "weights" / "theta" / "dras_theta"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_BASENAME.parent.mkdir(parents=True, exist_ok=True)


def find_latest_episode() -> int:
    pattern = re.compile(r"_policy_(\d+)\.weights\.h5$")
    latest = 0
    for f in WEIGHT_BASENAME.parent.glob(f"{WEIGHT_BASENAME.name}_policy_*.weights.h5"):
        m = pattern.search(f.name)
        if m:
            latest = max(latest, int(m.group(1)))
    return latest


def average_reward(rew_file: Path):
    if not rew_file.exists():
        return None
    vals = []
    with rew_file.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    vals.append(float(line))
                except ValueError:
                    pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def run_cqsim(jobset: Path, episode: int, training: bool) -> Path:
    basename = jobset.stem
    job_rel = jobset.relative_to(PROJECT_ROOT)
    args = [
        "python", "cqsim.py",
        "--config_sys", CFG_FILE,
        "--job", str(job_rel),
        "--node", str(job_rel),
        "--weight_num", str(episode),
        "--is_training", "1" if training else "0",
        "--output", basename,
        "--debug", f"debug_{basename}",
        "--path_in", "../",
        "--path_fmt", "../",
        "--path_out", "results/theta/",
        "--path_debug", "debug/theta/",
        "--debug_lvl", "10",
    ]
    subprocess.run(args, cwd=SRC_DIR, check=True)
    return RESULT_DIR / f"{basename}.rew"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rl-scheduler with resume capability")
    parser.add_argument("--validation_interval", type=int, default=5, help="validate every N episodes")
    args = parser.parse_args()

    jobsets = (
        sorted((JOBSET_DIR / "sampled").glob("*.swf")) +
        sorted((JOBSET_DIR / "real").glob("*.swf")) +
        sorted((JOBSET_DIR / "synthetic").glob("*.swf"))
    )

    latest = find_latest_episode()
    print(f"Latest completed episode: {latest}")

    for idx, jobset in enumerate(jobsets[latest:], start=latest + 1):
        print(f"=== Episode {idx} ({jobset.stem}) ===")
        rew = run_cqsim(jobset, idx, training=True)
        avg = average_reward(rew)
        if avg is not None:
            print(f"Episode {idx} average reward: {avg:.4f}")
        else:
            print(f"Episode {idx} reward file missing")

        if idx % args.validation_interval == 0:
            print(f"--- validation after episode {idx} ---")
            val_job = JOBSET_DIR / "validation_2023_jan.swf"
            v_rew = run_cqsim(val_job, idx, training=False)
            v_avg = average_reward(v_rew)
            if v_avg is not None:
                print(f"Validation average reward: {v_avg:.4f}")
            else:
                print("Validation reward file missing")


if __name__ == "__main__":
    main()
