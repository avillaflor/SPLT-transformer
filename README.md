## Installation

All python dependencies are in [`environment.yml`](environment.yml). Install with:

```
conda env create -f environment.yml
conda activate splt_transformer
```

### Installing CARLA

Install CARLA v0.9.11 (https://carla.org/2020/12/22/release-0.9.11/) for which the binaries are available here: (https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz)

```
mkdir $HOME/carla911
cd $HOME/carla911
wget "https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz"
tar xvzf CARLA_0.9.11.tar.gz
```

Add the following to your `.bashrc`:

```
export CARLA_9_11_PATH=$HOME/carla911
export CARLA_9_11_PYTHONPATH=$CARLA_9_11_PATH/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
```

### Installing Mujoco

In order to run the D4RL experiments, you will need to install mujoco. Please refer to [mujoco_py](https://github.com/openai/mujoco-py) for details.

## Usage
### Toy Car Problem (Ours)

```
python scripts/splt/bt/toycar_train.py --device cuda:0 --exp_name name
```

```
python scripts/splt/bt/toycar_plan.py --device cuda:0 --gpt_loadpath name
```

### Modified NoCrash

```
python scripts/splt/bt/nocrash_train.py --device cuda:0 --exp_name name
```

```
python scripts/splt/bt/nocrash_plan.py --device cuda:0 --gpt_loadpath name
```

### D4RL

```
python scripts/splt/bt/d4rl_train.py --dataset halfcheetah-medium-expert-v2 \
	--device cuda:0 --exp_name name
```

```
python scripts/splt/bt/d4rl_plan.py --dataset halfcheetah-medium-expert-v2 \
	--device cuda:0 --gpt_loadpath name
```

## Acknowledgements

Heavily based off of Trajectory Transformer [TT](https://github.com/JannerM/trajectory-transformer) repo.

IQL implementation based off of [rlkit](https://github.com/rail-berkeley/rlkit) repo.

DT implementation based off of Decision Transformer [DT](https://github.com/kzl/decision-transformer) repo.
