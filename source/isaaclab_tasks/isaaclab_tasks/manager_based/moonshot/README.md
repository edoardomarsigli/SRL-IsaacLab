# SRL Moonshot - Isaac Lab 

This repository serves as the working space for Isaac Lab related tasks using the [Manager Based Reinforcement Learning Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html) format. It functions as an isolated unit that integrates into the Isaac Lab methodology, meaning that all SRL Moonshot related content is contained within this repository.


## The file structure

**`./descriptions/`** : This folder contains all robot description related files such as URDFs, meshes and USDs. It also includes a `/config/` folder that has robot specific configuration details such as joint and actuator parameters used for Isaac Lab (see `moonbot_cfgs.py`) or terrains used in the environments.

**`./utils/`** : This folder contains utility functions related to integrating this workspace into Isaac Lab such as path utilities helping to find the main "IsaacLab" directory.

**`./{EXAMPLE_TASK}/`**: The rest of the folders contain categories of tasks such as:
- **`./locomotion/`**: Learning locomotion with velocity commands
- **`./navigation/`**: Learning to navigate with target commands
- **`./manipulation/`**: Learning to manipulate objects 

These folders each contain custom reward, event, and termination functions, curriculums, terrain configurations and more that relate to the task at hand. Currently, only locomotion tasks have been implemented. If you wish to implement other tasks, it would make sense to model them after the similar task directories in the `manager_based` directory. 

### *Where are the robots at???*
Each of the task folders contain a folder related to each specific robot in the `config`, e.g. `/hero_vehicle/`, `/hero_dragon/`, etc. This means that you may find duplicates of the robot folders but located within separate task folders.

Each robot folder then has its own agent (read: RL algorithm) configurations for the different workflows (read: collection of agents) such as [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) or [RSL-RL](https://github.com/leggedrobotics/rsl_rl).  
