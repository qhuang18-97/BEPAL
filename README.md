# Code of BEPAL for AAAI reproducibility Check
## Installation

First, clone the repo as BEPAL and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```bash
cd BEPAL/ic3net-envs
python setup.py develop
```
Next, install dependencies:
```bash
pip install -r requirements.txt
```

## Training
Training args for Predator Prey moving prey:
```bash
python main.py --env_name predator_prey --batch_size 500 --nagents 5 --nprocesses 1 --num_epochs 4000 --hid_size 64 --detach_gap 10 --lrate 0.001 --dim 12 --max_steps 40 --ic3net --vision 2 --obstacles 10 --recurrent --mode cooperative 
```
remove `--moving_prey` for difficulty level 1 and 2. Adjust `--nagents, --dim, --max_steps` for different map settings. For difficulty level 3 and 4, go to `ic3net-envs/ic3net_envs/predator_prey_env.py` and comment upper escape function for difficulty level 4, comment lower escape function for difficulty level 3. 

Training args for Traffic Junction:
```bash
python main.py --env_name traffic_junction --nagents 20 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 18 --max_steps 80 --ic3net --vision 1 --recurrent --add_rate_min 0.1 --add_rate_max 0.1 --curr_start 0 --curr_end 0 --difficulty hard
```
Adjust `add_rate_min add_rate_max` to 0.1 or 0.2 for different car add rate. 
