# Decision Stacks: Flexible Reinforcement Learning via Modular Generative Models

[[Website]](https://siyan-zhao.github.io/decision-stacks/)
[[Paper]](https://siyan-zhao.github.io/decision-stacks/)

Authors: Siyan Zhao, Aditya Grover

Reinforcement learning provides a compelling approach for tackling various aspects of sequential decision making, such as defining complex goals, planning future actions and observations, and evaluating their utilities. However, effectively integrating these capabilities while maintaining both expressive power and flexibility in modeling choices poses significant algorithmic challenges for efficient learning and inference. In this work, we introduce Decision Stacks, a generative framework that decomposes goal-conditioned policy agents into three distinct generative modules. These modules utilize independent generative models to simulate the temporal evolution of observations, rewards, and actions, enabling parallel learning through teacher forcing. Our framework ensures both expressivity and flexibility by allowing designers to tailor individual modules to incorporate architectural bias, optimization objectives, dynamics, domain transferability, and inference speed. Through extensive empirical evaluations, we demonstrate the effectiveness of Decision Stacks in offline policy optimization across various Markov Decision Processes (MDPs) and Partially Observable Markov Decision Processes (POMDPs), outperforming existing methods and facilitating flexible generative decision making.

![Example Trajectory](https://github.com/siyan-zhao/decision-stacks/blob/main/resources/traj.gif)
![Decision Stacks Framework](https://github.com/siyan-zhao/decision-stacks/blob/main/resources/ds.gif)

## Code and Instructions

## Environment Installation

To set up the environment required for running the project, follow the steps below:

1. Clone the repository:
Dependencies are in [`env.yml`](env.yml). Install with:

```
conda env create -f env.yml
conda activate decisionstacks
pip install -e .
```

2. Install DR4L 
  ```
  git clone https://github.com/Farama-Foundation/d4rl.git
  cd d4rl
  pip install -e .
  
  ```
  Or, alternatively:
  ```
  pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
  ```

We will be releasing the instructions shortly. Stay tuned for updates!

