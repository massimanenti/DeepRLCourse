"""
Defines a pytorch policy as the agent's actor

Functions to edit:
    2. forward
    3. update
"""

import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(nn.Tanh())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))

    mlp = nn.Sequential(*layers)
    # nn.Sequential is a module in PyTorch that allows you to build 
    # a neural network model where layers are arranged in a linear sequence.
    # The *layers syntax unpacks the list so that each element of the list is passed as a separate argument to nn.Sequential.
    return mlp


class MLPPolicySL(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    """
    Defines an MLP for supervised learning which maps observations to continuous
    actions.

    Attributes
    ----------
    mean_net: nn.Sequential
        A neural network that outputs the mean for continuous actions
    logstd: nn.Parameter
        A separate parameter to learn the standard deviation of actions

    Methods
    -------
    forward:
        Runs a differentiable forwards pass through the network
    update:
        Trains the policy with a supervised learning objective
    """
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        # self.mean_net will be a neural network that outputs the mean of the actions
        self.mean_net = build_mlp(
            input_size=self.ob_dim,
            output_size=self.ac_dim,
            n_layers=self.n_layers, size=self.size,
        )
        self.mean_net.to(ptu.device)
        # recall that nn.Parameter(...) generates a learnable parameter. In this case it will probably
        # represent the logarithm of the standard deviation of the policy.
        self.logstd = nn.Parameter(

            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.logstd.to(ptu.device)
        self.optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()),
            self.learning_rate
        )

    def save(self, filepath):
        """
        :param filepath: path to save MLP
        """
        torch.save(self.state_dict(), filepath)

    def forward(self, observation: torch.FloatTensor) -> distributions.Normal:
        """
        Defines the forward pass of the network

        :param observation: observation(s) to query the policy
        :return:
            action: sampled action(s) from the policy
        """
        # TODO: implement the forward pass of the network.
        # You can return anything you want, but you should be able to differentiate
        # through it. For example, you can return a torch.FloatTensor. You can also
        # return more flexible objects, such as a
        # `torch.distributions.Distribution` object. It's up to you!

        mean = self.mean_net(observation)
        std = self.logstd.exp()

        return distributions.Normal(mean, std)

    def update(self, observations, actions):
        """
        Updates/trains the policy

        :param observations: observation(s) to query the policy (train_batch_size, n_states)
        :param actions: actions we want the policy to imitate (train_batch_size, n_actions)
        :return:
            dict: 'Training Loss': supervised learning loss
        """
        # TODO: update the policy and return the loss

        # Move data to the appropriate device and converts numpy array to torch tensor
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)

        # check dimension
        dim_obs_batch = observations.shape
        dim_act_batch = actions.shape
        assert dim_obs_batch[0] == dim_act_batch[0], 'Error: the batch size of the observation array is different from the action one'
        assert dim_obs_batch[1] == self.ob_dim, 'Error: observation dimension used in the batch not matching the expected one'
        assert dim_act_batch[1] == self.ac_dim, 'Error: action dimension used in the batch not matching the expected one'

        for curr_obs, curr_act in zip(observations, actions):
            # generate distribution of the learned policy
            action_distribution = self.forward(curr_obs)
            # Compute the loss as the negative log likelihood
            loss = -action_distribution.log_prob(curr_act).sum()
        
            # Perform a gradient descent step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
