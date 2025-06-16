# RL HPC Scheduler

This project contains tools to train and evaluate a reinforcement-learning based job scheduler using traces from the Theta supercomputer. Jobsets are provided in the `data/jobsets` directory in Standard Workload Format (SWF).

## Requirements
- Python 3.8+
- TensorFlow 2
- numpy, pandas and other common scientific packages

Install dependencies with pip:
```bash
pip install tensorflow pandas numpy
```

## Training
Run the training script from the project root:
```bash
python train.py [--validation_interval N]
```
The script iterates over jobsets from `data/jobsets/{sampled,real,synthetic}`. For each episode it calls `src/cqsim.py` and saves weights to `weights/theta/dras_theta_policy_<episode>.weights.h5`.

If weight files already exist the script resumes from the latest episode. Every `N` episodes (default 5) it runs the model on `data/jobsets/validation_2023_jan.swf` and prints the average reward from the generated `.rew` files so you can track convergence. During training batches the simulator also prints the batch loss so you can monitor learning progress.

## Validation
To evaluate a range of weights later you can use the legacy bash script from inside `src`:
```bash
cd src
bash validate.bash
```
Results are written to `data/results/theta` and debug logs to `data/debug/theta`.

## Configuration
Scheduler parameters are defined in `src/Config/config_sys.set`. Notable options:
- `reward_type` – selects the reward calculation:
  - `1` utilisation only
  - `2` wait-time only
  - `3` utilisation + wait-time
  - `4` utilisation with priority award
  - `5` node occupancy, wait time and job size (default)
  - `6` job size only
  - `7` carbon-aware reward
- `weight_name` – base name for the weight files
- `weight_num` – starting episode when running in evaluation mode
- `is_training` – `1` for training, `0` for inference

## Data
Training and validation jobsets are SWF files containing one job per line after a small header. They can be generated from raw CSV traces using the tools in `src/DataPreparation/jobset_generator.py`. Each line includes submit time, run time, resource requests and a final carbon consideration index used by the carbon-aware reward.

Place generated jobsets in the respective subfolders of `data/jobsets` before running `train.py`.

