#!/usr/bin/env python3
"""Basic sanity-tests for CarbonIntensityGrid.carbon_reward
Run:  python -m CqSim.carbon_intensity_grid_test
(or simply `python src/CqSim/carbon_intensity_grid_test.py` from the repo root)
"""
import datetime
from CqSim.carbon_intensity_grid import CarbonIntensityGrid


def pretty(ts):
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def run_case(start_time, reserved_sec, cpus=32, idx=0.5):
    grid = CarbonIntensityGrid(forecast_type="daily")
    reward = grid.carbon_reward(start_time, reserved_sec, carbon_consideration_index=idx, numProcess=cpus)
    print(f"Start: {pretty(start_time)}  Duration: {reserved_sec/3600:.2f} h  CPUs: {cpus} --> reward = {reward:.2f}\n")


def main():
    # Case 1 – one-hour job starting at an arbitrary time on the same day
    start = datetime.datetime(2022, 6, 6, 10, 15, 0, 100000)  # 10:15:00.1 UTC
    run_case(start, reserved_sec=3600)

    # Case 2 – 90-minute job crossing an hour boundary
    start = datetime.datetime(2022, 6, 6, 22, 45, 0, 500000)  # 22:45:00.5 UTC
    run_case(start, reserved_sec=5400)

    # Case 3 – job that crosses midnight into the next day
    start = datetime.datetime(2022, 6, 6, 23, 30, 0, 200000)  # 23:30:00.2 UTC
    run_case(start, reserved_sec=7200)

    # Case 4 – long multi-day job (30 h)
    start = datetime.datetime(2022, 6, 6, 6, 0, 0, 700000)  # 06:00:00.7 UTC
    run_case(start, reserved_sec=30*3600)


if __name__ == "__main__":
    main()
