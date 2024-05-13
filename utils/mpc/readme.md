# MPC Solver

The `BaseMpcSolver` class is wrapper for Nonlinear Programming (NLP) solvers that can be used to solve Optimal Control Problems (OCP). You can subclass the `BaseMpcSolver` class to create a wrapper of the actually OCP solver, like *acados* or *ipopt*.


# MPC Layer

The MPC layer extends the autograd engine in PyTorch by implementing a custom function which relies on non-PyTorch libraries to solve Optimal Control Problems.

Please notice that, the current implementation of `MpcLayer` does NOT contain any learnable parameters.

Useful resources:
- [Tutorials on extending `torch.autograd`](https://pytorch.org/docs/stable/notes/extending.html)
- [Some example code pieces from `cvxpy`](https://github.com/cvxgrp/cvxpylayers/blob/master/cvxpylayers/torch/cvxpylayer.py)

# MPC Policy
 MPC can be a policy.

`MpcPolicy`: The policy is given implicitly by calling the MPC solver to solve the underlying OCP.

`ActorCriticMpcPolicy`: Use the optimal action given by MPC as actor, the optimal value function given by MPC as critic.

# MPC Network

The MPC network is a neural network containing a MPC layer. We follow the same architecture as [Actor-Critic Model Predictive Control]().
