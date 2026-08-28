# Code of BEPAL for AAAI reproducibility Check

Supported environments (`--env_name`): `predator_prey`, `traffic_junction`,
`starcraft`, `rware`.

## Installation

> **pip must be older than 24.1.** `gym==0.21.0` ships malformed metadata
> (`opencv-python (>=3.)`), and pip 24.1 onwards refuses to install it *or*
> anything that depends on it, failing with
> `InvalidRequirement: Expected matching RIGHT_PARENTHESIS`. This is a defect in
> gym, not in this repo, and there is no way around it other than:
> ```bash
> pip install "pip<24.1"
> ```

Each environment lives in its own installable package under `envs/`, and each
declares its own dependencies, so you only need to install the ones you plan to
train on. `data.py` imports them lazily, so the other environments still run
when their packages are absent.

Use a **separate conda environment per environment family**, since their
dependency sets differ. Each yml pins `pip` below 24.1 for the reason above:

```bash
# Predator-Prey and Traffic-Junction
conda env create -f environment_ic3net.yml   # creates an env named IC3NET
conda activate IC3NET
pip install -e envs/ic3net-envs

# RWARE warehouse
conda env create -f environment_rware.yml    # creates an env named RWARE
conda activate RWARE
pip install -e envs/rware-envs
```

Each `envs/*/setup.py` declares its own pinned dependencies (`torch`, `gym`,
`numpy`, `visdom`, and `rware` for the warehouse), so if you already have a
suitable interpreter with `pip<24.1` you can skip the conda step and just run
the `pip install -e` line for the environment you want.

## Training

Training args for Predator Prey moving prey:
```bash
python main.py --env_name predator_prey --batch_size 500 --nagents 5 --nprocesses 1 --num_epochs 4000 --hid_size 64 --detach_gap 10 --lrate 0.001 --dim 12 --max_steps 40 --ic3net --vision 2 --obstacles 10 --recurrent --mode cooperative 
```
remove `--moving_prey` for difficulty level 1 and 2. Adjust `--nagents, --dim, --max_steps` for different map settings. For difficulty level 3 and 4, go to `envs/ic3net-envs/ic3net_envs/predator_prey_env.py` and comment upper escape function for difficulty level 4, comment lower escape function for difficulty level 3. 

Training args for Traffic Junction:
```bash
python main.py --env_name traffic_junction --nagents 20 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 18 --max_steps 80 --ic3net --vision 1 --recurrent --add_rate_min 0.1 --add_rate_max 0.1 --curr_start 0 --curr_end 0 --difficulty hard
```
Adjust `add_rate_min add_rate_max` to 0.1 or 0.2 for different car add rate. 

Training args for RWARE (robotic warehouse):
```bash
python main.py --use_teacher --env_name rware --batch_size 500 --nagents 4 \
    --nprocesses 1 --num_epochs 8001 --hid_size 64 --detach_gap 10 \
    --lrate 0.003 --max_steps 500 --gamma 0.99 --ic3net --recurrent \
    --save_every 100 --save results/
```
`--nagents` must be `4`: the layout is hardcoded to `rware-tiny-4ag-v1` with
`request_queue_size=4` in `envs/rware-envs/rware_envs/rware_env.py`. 
