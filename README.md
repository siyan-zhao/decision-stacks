# Decision Stacks: Flexible Reinforcement Learning via Modular Generative Models

[[Website]](https://siyan-zhao.github.io/decision-stacks/)
[[Paper]]
Authors: Siyan Zhao, Aditya Grover

Reinforcement learning provides a compelling approach for tackling various aspects of sequential decision making, such as defining complex goals, planning future actions and observations, and evaluating their utilities. However, effectively integrating these capabilities while maintaining both expressive power and flexibility in modeling choices poses significant algorithmic challenges for efficient learning and inference. In this work, we introduce Decision Stacks, a generative framework that decomposes goal-conditioned policy agents into three distinct generative modules. These modules utilize independent generative models to simulate the temporal evolution of observations, rewards, and actions, enabling parallel learning through teacher forcing. Our framework ensures both expressivity and flexibility by allowing designers to tailor individual modules to incorporate architectural bias, optimization objectives, dynamics, domain transferability, and inference speed. Through extensive empirical evaluations, we demonstrate the effectiveness of Decision Stacks in offline policy optimization across various Markov Decision Processes (MDPs) and Partially Observable Markov Decision Processes (POMDPs), outperforming existing methods and facilitating flexible generative decision making.

![Example Trajectory](https://github.com/siyan-zhao/decision-stacks/blob/main/resources/traj.gif)
![Decision Stacks Framework](https://github.com/siyan-zhao/decision-stacks/blob/main/resources/ds.gif)

## Code and Instructions

We will be releasing the code and instructions shortly. Stay tuned for updates!
