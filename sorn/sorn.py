# -*- coding: utf-8 -*-
# MIT License

# Copyright (c) 2019 Saranraj Nambusubramaniyan

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division
import numpy as np
import os
import random
import logging
from multiprocessing import Pool
from tqdm import tqdm
from sorn.callbacks import *
from sorn.utils import Initializer, generate_exp


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    handlers=[logging.FileHandler("sorn.log"), logging.StreamHandler()],
)


class Sorn(object):

    """This class wraps initialization of the network and its parameters"""

    nu = 10
    ne = 200
    ni = int(0.2 * ne)
    eta_stdp = 0.004
    eta_inhib = 0.001
    eta_ip = 0.01
    te_max = 1.0
    ti_max = 0.5
    ti_min = 0.0
    te_min = 0.0
    mu_ip = 0.1
    sigma_ip = 0.0
    network_type_ee = "Sparse"
    network_type_ei = "Sparse"
    network_type_ie = "Dense"
    lambda_ee = 20
    lambda_ei = 20
    lambda_ie = 100

    @staticmethod
    def initialize_weight_matrix(
        network_type: str, synaptic_connection: str, self_connection: str, lambd_w: int
    ):
        """Wrapper for initializing the weight state for SORN

        Args:
            network_type (str): Spare or Dense

            synaptic_connection (str): EE,EI,IE. Note that Spare connection is defined only for EE connections

            self_connection (str): True or False: Synaptic delay or time delay

            lambd_w (int): Average number of incoming and outgoing connections per neuron

        Returns:
            weight_matrix (array): Array of connection strengths
        """

        if (network_type == "Sparse") and (self_connection == "False"):

            # Generate weight matrix for E-E/ E-I connections with mean lamda incoming and out-going connections per neuron
            assert (
                lambd_w <= Sorn.ne
            ), "Number of connections per unit (lambda) should be less than number of units(Ne) in the pool and also Ne should be greater than 25"
            weight_matrix = Initializer.generate_lambd_connections(
                synaptic_connection, Sorn.ne, Sorn.ni, lambd_w, lambd_std=1
            )

        # Dense matrix for W_ie
        elif (network_type == "Dense") and (self_connection == "False"):

            # Uniform distribution of weights
            weight_matrix = np.random.uniform(0.0, 0.1, (Sorn.ne, Sorn.ni))
            weight_matrix.reshape((Sorn.ne, Sorn.ni))

        return weight_matrix

    @staticmethod
    def initialize_threshold_matrix(
        te_min: float, te_max: float, ti_min: float, ti_max: float
    ):
        """Initialize the threshold for excitatory and inhibitory neurons

        Args:
            te_min (float): Min threshold value for excitatory units
            te_max (float): Min threshold value for inhibitory units
            ti_min (float): Max threshold value for excitatory units
            ti_max (float): Max threshold value for inhibitory units

        Returns:
            te (array): Threshold values for excitatory units
            ti (array): Threshold values for inhibitory units
        """

        te = np.random.uniform(te_min, te_max, (Sorn.ne, 1))
        ti = np.random.uniform(ti_min, ti_max, (Sorn.ni, 1))

        return te, ti

    @staticmethod
    def initialize_activity_vector(ne: int, ni: int):
        """Initialize the activity vectors X and Y for excitatory and inhibitory neurons

        Args:
            ne (int): Number of excitatory neurons
            ni (int): Number of inhibitory neurons

        Returns:
            x (array): Array of activity vectors of excitatory population
            y (array): Array of activity vectors of inhibitory population"""

        x = np.zeros((ne, 2))
        y = np.zeros((ni, 2))

        return x, y


class Plasticity(Sorn):
    """Instance of class Sorn. Inherits the variables and functions defined in class Sorn.
    It encapsulates all plasticity mechanisms mentioned in the article. Inherits all attributed from parent class Sorn
    """

    def __init__(self):

        super().__init__()
        self.nu = Sorn.nu  # Number of input units
        self.ne = Sorn.ne  # Number of excitatory units
        self.eta_stdp = (
            Sorn.eta_stdp
        )  # STDP plasticity Learning rate constant; SORN1 and SORN2
        self.eta_ip = (
            Sorn.eta_ip
        )  # Intrinsic plasticity learning rate constant; SORN1 and SORN2
        self.eta_inhib = (
            Sorn.eta_inhib
        )  # Intrinsic plasticity learning rate constant; SORN2 only
        self.h_ip = 2 * Sorn.nu / Sorn.ne  # Target firing rate
        self.mu_ip = Sorn.mu_ip  # Mean target firing rate
        # Number of inhibitory units in the network
        self.ni = int(0.2 * Sorn.ne)
        self.timesteps = Sorn.timesteps  # Total time steps of simulation
        self.te_min = Sorn.te_min  # Excitatory minimum Threshold
        self.te_max = Sorn.te_max  # Excitatory maximum Threshold

    def stdp(self, wee: np.array, x: np.array, cutoff_weights: list):
        """Apply STDP rule : Regulates synaptic strength between the pre(Xj) and post(Xi) synaptic neurons
        Args:
            wee (array):  Weight matrix
            x (array): Excitatory network activity
            cutoff_weights (list): Maximum and minimum weight ranges
        Returns:
            wee (array):  Weight matrix
        """

        x = np.asarray(x)
        xt_1 = x[:, 0]
        xt = x[:, 1]
        wee_t = wee.copy()

        # STDP applies only on the neurons which are connected.

        for i in range(len(wee_t[0])):  # Each neuron i, Post-synaptic neuron

            for j in range(
                len(wee_t[0:])
            ):  # Incoming connection from jth pre-synaptic neuron to ith neuron

                if wee_t[j][i] != 0.0:  # Check connectivity

                    # Get the change in weight
                    delta_wee_t = self.eta_stdp * (xt[i] * xt_1[j] - xt_1[i] * xt[j])

                    # Update the weight between jth neuron to i ""Different from notation in article

                    wee_t[j][i] = wee[j][i] + delta_wee_t

        # Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight
        wee_t = Initializer.reset_min(wee_t, cutoff_weights[0])

        # Check and set all weights < upper cutoff weight
        wee_t = Initializer.reset_max(wee_t, cutoff_weights[1])

        return wee_t

    def ip(self, te: np.array, x: np.array):
        """Intrinsic Plasiticity mechanism
        Args:
            te (array): Threshold vector of excitatory units
            x (array): Excitatory network activity
        Returns:
            te (array): Threshold vector of excitatory units
        """

        # IP rule: Active unit increases its threshold and inactive decreases its threshold.
        xt = x[:, 1]

        te_update = te + self.eta_ip * (xt.reshape(self.ne, 1) - self.h_ip)

        # Check whether all te are in range [0.0,1.0] and update acordingly

        # Update te < 0.0 ---> 0.0
        te_update = Initializer.reset_min(te_update, self.te_min)

        # Set all te > 1.0 --> 1.0
        te_update = Initializer.reset_max(te_update, self.te_max)

        return te_update

    @staticmethod
    def ss(wee: np.array):
        """Synaptic Scaling or Synaptic Normalization
        Args:
            wee (array):  Weight matrix
        Returns:
            wee (array):  Scaled Weight matrix
        """
        wee = wee / np.sum(wee, axis=0)
        return wee

    def istdp(self, wei: np.array, x: np.array, y: np.array, cutoff_weights: list):
        """Apply iSTDP rule, which regulates synaptic strength between the pre inhibitory(Xj) and post Excitatory(Xi) synaptic neurons
        Args:
            wei (array): Synaptic strengths from inhibitory to excitatory
            x (array): Excitatory network activity
            y (array): Inhibitory network activity
            cutoff_weights (list): Maximum and minimum weight ranges
        Returns:
            wei (array): Synaptic strengths from inhibitory to excitatory"""

        # Excitatory network activity
        xt = np.asarray(x)[:, 1]

        # Inhibitory network activity
        yt_1 = np.asarray(y)[:, 0]

        # iSTDP applies only on the neurons which are connected.
        wei_t = wei.copy()

        for i in range(
            len(wei_t[0])
        ):  # Each neuron i, Post-synaptic neuron: means for each column;

            for j in range(
                len(wei_t[0:])
            ):  # Incoming connection from j, pre-synaptic neuron to ith neuron

                if wei_t[j][i] != 0.0:  # Check connectivity

                    # Get the change in weight
                    delta_wei_t = (
                        -self.eta_inhib * yt_1[j] * (1 - xt[i] * (1 + 1 / self.mu_ip))
                    )

                    # Update the weight between jth neuron to i ""Different from notation in article

                    wei_t[j][i] = wei[j][i] + delta_wei_t

        # Prune the smallest weights induced by plasticity mechanisms; Apply lower cutoff weight
        wei_t = Initializer.reset_min(wei_t, cutoff_weights[0])

        # Check and set all weights < upper cutoff weight
        wei_t = Initializer.reset_max(wei_t, cutoff_weights[1])

        return wei_t

    @staticmethod
    def structural_plasticity(wee: np.array):
        """Add new connection value to the smallest weight between excitatory units randomly
        Args:
            wee (array): Weight matrix
        Returns:
            wee (array):  Weight matrix"""

        p_c = np.random.randint(0, 10, 1)

        if p_c == 0:  # p_c= 0.1

            # Do structural plasticity
            # Choose the smallest weights randomly from the weight matrix wee
            indexes = Initializer.get_unconnected_indexes(wee)

            # Choose any idx randomly such that i!=j
            while True:
                idx_rand = random.choice(indexes)
                if idx_rand[0] != idx_rand[1]:
                    break

            wee[idx_rand[0]][idx_rand[1]] = 0.001

        return wee

    @staticmethod
    def initialize_plasticity():
        """Initialize weight state for plasticity phase based on network configuration

        Args:
            kwargs (self.__dict__): All arguments are inherited from Sorn attributes

        Returns:
            tuple(array): Weight state WEI, WEE, WIE and threshold state Te, Ti and Initial state vectors X,Y"""

        sorn_init = Sorn()
        WEE_init = sorn_init.initialize_weight_matrix(
            network_type=Sorn.network_type_ee,
            synaptic_connection="EE",
            self_connection="False",
            lambd_w=Sorn.lambda_ee,
        )
        WEI_init = sorn_init.initialize_weight_matrix(
            network_type=Sorn.network_type_ei,
            synaptic_connection="EI",
            self_connection="False",
            lambd_w=Sorn.lambda_ei,
        )
        WIE_init = sorn_init.initialize_weight_matrix(
            network_type=Sorn.network_type_ie,
            synaptic_connection="IE",
            self_connection="False",
            lambd_w=Sorn.lambda_ie,
        )

        Wee_init = Initializer.zero_sum_incoming_check(WEE_init)
        Wei_init = Initializer.zero_sum_incoming_check(WEI_init)
        Wie_init = Initializer.zero_sum_incoming_check(WIE_init)

        c = np.count_nonzero(Wee_init)
        v = np.count_nonzero(Wei_init)
        b = np.count_nonzero(Wie_init)

        logging.info("Network Initialized")
        logging.info("Number of connections in Wee %s , Wei %s, Wie %s" % (c, v, b))
        logging.info(
            "Shapes Wee %s Wei %s Wie %s"
            % (Wee_init.shape, Wei_init.shape, Wie_init.shape)
        )

        # Normalize the incoming weights

        normalized_wee = Initializer.normalize_weight_matrix(Wee_init)
        normalized_wei = Initializer.normalize_weight_matrix(Wei_init)
        normalized_wie = Initializer.normalize_weight_matrix(Wie_init)

        te_init, ti_init = sorn_init.initialize_threshold_matrix(
            Sorn.te_min, Sorn.te_max, Sorn.ti_min, Sorn.ti_max
        )
        x_init, y_init = sorn_init.initialize_activity_vector(Sorn.ne, Sorn.ni)

        # Initializing variables from sorn_initialize.py

        wee = normalized_wee.copy()
        wei = normalized_wei.copy()
        wie = normalized_wie.copy()
        te = te_init.copy()
        ti = ti_init.copy()
        x = x_init.copy()
        y = y_init.copy()

        return wee, wei, wie, te, ti, x, y


class MatrixCollection(Sorn):
    """Collect all state initialized and updated during simulation(plasiticity and training phases)

    Args:
        phase(str): Training or Plasticity phase

        state(dict): Network activity, threshold and connection state

    Returns:
        MatrixCollection instance"""

    def __init__(self, phase, state=None):
        super().__init__()

        self.phase = phase
        self.state = state
        if self.phase == "plasticity" and self.state == None:

            self.timesteps = Sorn.timesteps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = (
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
            )
            wee, wei, wie, te, ti, x, y = Plasticity.initialize_plasticity()

            # Assign initial matrix to the master state
            self.Wee[0] = wee
            self.Wei[0] = wei
            self.Wie[0] = wie
            self.Te[0] = te
            self.Ti[0] = ti
            self.X[0] = x
            self.Y[0] = y

        elif self.phase == "plasticity" and self.state != None:

            self.timesteps = Sorn.timesteps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = (
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
            )
            # Assign state from plasticity phase to the new master state for training phase
            self.Wee[0] = state["Wee"]
            self.Wei[0] = state["Wei"]
            self.Wie[0] = state["Wie"]
            self.Te[0] = state["Te"]
            self.Ti[0] = state["Ti"]
            self.X[0] = state["X"]
            self.Y[0] = state["Y"]

        elif self.phase == "training":

            # NOTE:timesteps here is diferent for plasticity and training phase
            self.timesteps = Sorn.timesteps + 1  # Total training steps
            self.Wee, self.Wei, self.Wie, self.Te, self.Ti, self.X, self.Y = (
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
                [0] * self.timesteps,
            )
            # Assign state from plasticity phase to new respective state for training phase
            self.Wee[0] = state["Wee"]
            self.Wei[0] = state["Wei"]
            self.Wie[0] = state["Wie"]
            self.Te[0] = state["Te"]
            self.Ti[0] = state["Ti"]
            self.X[0] = state["X"]
            self.Y[0] = state["Y"]

    def weight_matrix(self, wee: np.array, wei: np.array, wie: np.array, i: int):
        """Update weight state

        Args:
            wee(array): Excitatory-Excitatory weight matrix

            wei(array): Inhibitory-Excitatory weight matrix

            wie(array): Excitatory-Inhibitory weight matrix

            i(int): Time step

        Returns:
            tuple(array): Weight state Wee, Wei, Wie"""

        self.Wee[i + 1] = wee
        self.Wei[i + 1] = wei
        self.Wie[i + 1] = wie

        return self.Wee, self.Wei, self.Wie

    def threshold_matrix(self, te: np.array, ti: np.array, i: int):
        """Update threshold state

        Args:
            te(array): Excitatory threshold

            ti(array): Inhibitory threshold

            i(int): Time step

        Returns:
            tuple(array): Threshold state Te and Ti"""

        self.Te[i + 1] = te
        self.Ti[i + 1] = ti
        return self.Te, self.Ti

    def network_activity_t(
        self, excitatory_net: np.array, inhibitory_net: np.array, i: int
    ):
        """Network state at current time step

        Args:
            excitatory_net(array): Excitatory network activity

            inhibitory_net(array): Inhibitory network activity

            i(int): Time step

        Returns:
            tuple(array): Updated Excitatory and Inhibitory states
        """

        self.X[i + 1] = excitatory_net
        self.Y[i + 1] = inhibitory_net

        return self.X, self.Y

    def network_activity_t_1(self, x: np.array, y: np.array, i: int):
        """Network activity at previous time step

        Args:
            x(array): Excitatory network activity

            y(array): Inhibitory network activity

            i(int): Time step

        Returns:
            tuple(array): Previous Excitatory and Inhibitory states
        """
        x_1, y_1 = [0] * self.timesteps, [0] * self.timesteps
        x_1[i] = x
        y_1[i] = y

        return x_1, y_1


class Neurogenesis(Plasticity):
    """
    Module implements the neurogenesis in Excitatory and Inhibitory Pool
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.excitatory_neurogenesis = True
        self.inhibitory_neurogenesis = True

    def sample_weights(self, lambd=Sorn.lambda_ee):
        """_summary_

        Args:
            lambd (_type_, optional): _description_. Defaults to Sorn.lambda_ee.

        Returns:
            _type_: _description_
        """

        return np.random.uniform(0.0, 0.1, lambd)

    def sample_indices(self, pool, weights, synapse="ee", lambd=Sorn.lambda_ee):
        """_summary_

        Args:
            pool (_type_): _description_
            weights (_type_): _description_
            synapse (str, optional): _description_. Defaults to "ee".
            lambd (_type_, optional): _description_. Defaults to Sorn.lambda_ee.

        Returns:
            _type_: _description_
        """

        if pool == "excitatory":

            if synapse == "ee":
                assert lambd < weights.shape[1]
                indices = random.sample(list(range(weights.shape[1])), lambd)
            elif synapse == "ei":
                assert lambd <= weights.shape[0]
                indices = random.sample(list(range(0, weights.shape[0])), lambd)

            else:
                # synapse == "ie":
                assert lambd == weights.shape[1]  # Dense connection
                indices = random.sample(list(range(weights.shape[1])), lambd)

        if pool == "inhibitory":

            if synapse == "ei":
                assert lambd <= weights.shape[1]
                indices = random.sample(list(range(weights.shape[1])), lambd)
            else:
                # synapse == "ie":
                assert lambd == weights.shape[0]  # Dense connection

                indices = random.sample(list(range(weights.shape[0])), lambd)
        return indices

    def excitatory(self, wee, wei, wie):
        """_summary_

        Args:
            wee (_type_): _description_
            wei (_type_): _description_
            wie (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        # Excitatory pool
        # Choose incoming and outgoing synapses randomly
        eff_idxs_exc = self.sample_indices(
            pool="excitatory", weights=wee, synapse="ee", lambd=Sorn.lambda_ee
        )
        aff_idxs_exc = self.sample_indices(
            pool="excitatory", weights=wee, synapse="ee", lambd=Sorn.lambda_ee
        )
        # Sample incoming and outgoing synaptic weights
        eff_synapses = self.sample_weights(Sorn.lambda_ee)
        aff_synapses = self.sample_weights(Sorn.lambda_ee)

        # Apppend additional rows (outgoing synapses) and cols (incoming)
        temp_ee = np.zeros(np.array(wee.shape) + 1)
        temp_ee[: wee.shape[0], : wee.shape[1]] = wee  # Padding
        # Update outgoing synapses
        for eff_idx, w in zip(eff_idxs_exc, eff_synapses):
            temp_ee[-1][eff_idxs_exc] = w
        # Update incoming synapses
        for aff_idx, w in zip(aff_idxs_exc, aff_synapses):
            temp_ee[aff_idx][-1] = w

        # Afferent inhibitory connections to the neuron
        # aff_idxs_inh = self.sample_indices(wei, synapse="ei", lambd=Sorn.lambda_ei)
        aff_idxs_inh = self.sample_indices(
            pool="excitatory", weights=wei, synapse="ei", lambd=5
        )
        assert (
            max(aff_idxs_inh) <= Sorn.ni
        ), f"Max afferent inhibitory index is {max(aff_idxs_inh)}"

        aff_synapses_inh = self.sample_weights(
            lambd=len(aff_idxs_inh)
        )  # 40% connectivity. I-> E Eg. Wei.shape=[40,200]
        assert len(aff_idxs_inh) == len(aff_synapses_inh), "Synapses size mismatch"

        temp_ei = np.zeros((wei.shape[0], wei.shape[1] + 1))  # [40,201]
        temp_ei[: wei.shape[0], : wei.shape[1]] = wei  # Padding
        # Update outgoing synapses Wei: sparse inh -> exc synapse
        for idx, w in zip(aff_idxs_inh, aff_synapses_inh):
            temp_ei[idx][-1] = w

        # Excitatory --> Inhibitory

        # Efferent connections to Inhibitory Pool; Eci->Inh Dense connection
        eff_idxs_inh = self.sample_indices(
            pool="excitatory", weights=wie, synapse="ie", lambd=Sorn.ni
        )

        # Sample outgoing inhibitory synapses and outgoing excitatory synapses
        eff_synapses_inh = self.sample_weights(
            lambd=len(eff_idxs_inh)
        )  # 100% connectivity. E-> I Eg. Wie.shape=[200,40]
        assert len(eff_idxs_inh) == len(eff_synapses_inh), "Synapses size mismatch"
        temp_ie = np.zeros((wie.shape[0] + 1, wie.shape[1]))  # [201,40]
        temp_ie[: wie.shape[0], : wie.shape[1]] = wie  # Padding
        # Update outgoing synapses Wei: sparse inh -> exc synapse
        temp_ie[-1] = eff_synapses_inh

        Sorn.ne += 1

        return temp_ee, temp_ei, temp_ie

    def set_threshold(self, thresh):
        """_summary_

        Args:
            thresh (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Initial threshold value for the new neuron
        thresh = np.append(thresh, random.uniform(0.0, 0.1))[:, None]
        return thresh

    def inhibitory(self, wei, wie):

        """_summary_

        Returns:
            _type_: _description_
        """

        # Afferent inhibitory connections to the neuron
        # TODO: Lamdas are global parameters.
        # Check and set lamdas for local connection according to the synapse
        eff_idxs_exc = self.sample_indices(
            pool="inhibitory", weights=wei, synapse="ei", lambd=Sorn.lambda_ei
        )
        assert (
            max(eff_idxs_exc) < Sorn.ne
        ), f"Max afferent inhibitory index is {max(eff_idxs_exc)}"

        eff_synapses_exc = self.sample_weights(
            lambd=len(eff_idxs_exc)
        )  # 40% connectivity. I-> E Eg. Wei.shape=[40,200]
        assert len(eff_idxs_exc) == len(eff_synapses_exc), "Synapses size mismatch"

        temp_ei = np.zeros((wei.shape[0] + 1, wei.shape[1]))  # [40,201]
        temp_ei[: wei.shape[0], : wei.shape[1]] = wei  # Padding
        # Update outgoing synapses Wei: sparse inh -> exc synapse
        for idx, w in zip(eff_idxs_exc, eff_synapses_exc):
            temp_ei[-1][idx] = w

        # Excitatory --> Inhibitory
        # Afferent connections to Inhibitory Pool; Exc->Inh Dense connection
        aff_idxs_exc = self.sample_indices(
            pool="inhibitory", weights=wie, synapse="ie", lambd=Sorn.ne
        )

        # Sample outgoing inhibitory synapses and outgoing excitatory synapses
        aff_synapses_exc = self.sample_weights(
            lambd=len(aff_idxs_exc)
        )  # 100% connectivity. E-> I Eg. Wie.shape=[200,40]
        assert len(aff_idxs_exc) == len(aff_synapses_exc), "Synapses size mismatch"
        temp_ie = np.zeros((wie.shape[0], wie.shape[1] + 1))  # [201,40]
        temp_ie[: wie.shape[0], : wie.shape[1]] = wie  # Padding
        # Update outgoing synapses Wei: sparse inh -> exc synapse
        temp_ie[:, -1] = aff_synapses_exc

        Sorn.ni += 1

        return temp_ei, temp_ie

    def step(self, exc_pool, inh_pool, x_buffer, y_buffer, wee, wei, wie, te, ti):

        if exc_pool:
            self.excitatory_neurogenesis = exc_pool
            # Set initial state of new exc neuron
            x_temp = np.zeros(np.array(x_buffer.shape) + 1)
            x_temp[: x_buffer.shape[0], : x_buffer.shape[1]] = x_buffer
            x_buffer = x_temp.copy()

            # Set efferent and afferent synapses and threshold
            wee, wei, wie = self.excitatory(wee, wei, wie)
            te = self.set_threshold(thresh=te)

        if inh_pool:
            self.inhibitory_neurogenesis = inh_pool

            # Set initial state of new inh neuron
            y_temp = np.zeros(np.array(y_buffer.shape) + 1)
            y_temp[: y_buffer.shape[0], : y_buffer.shape[1]] = y_buffer
            y_buffer = y_temp.copy()

            # Set efferent and afferent synapses and threshold
            wei, wie = self.inhibitory(wei, wie)
            ti = self.set_threshold(ti)

        return x_buffer, y_buffer, wee, wei, wie, te, ti


class NetworkState(Plasticity):

    """The evolution of network states

    Args:
        v_t(array): External input/stimuli

    Returns:
        instance(object): NetworkState instance"""

    def __init__(self, v_t: np.array):
        super().__init__()
        self.v_t = v_t
        assert Sorn.nu == len(
            self.v_t
        ), "Input units and input size mismatch: {} != {}".format(
            Sorn.nu, len(self.v_t)
        )
        if Sorn.nu != Sorn.ne:
            self.v_t = list(self.v_t) + [0.0] * (Sorn.ne - Sorn.nu)
        self.v_t = np.expand_dims(self.v_t, 1)

    def incoming_drive(self, weights: np.array, activity_vector: np.array):
        """Excitatory Post synaptic potential towards neurons in the reservoir in the absence of external input

        Args:
            weights(array): Synaptic strengths

            activity_vector(list): Acitivity of inhibitory or Excitatory neurons

        Returns:
            incoming(array): Excitatory Post synaptic potential towards neurons
        """
        incoming = weights * activity_vector
        incoming = np.array(incoming.sum(axis=0))
        return incoming

    def excitatory_network_state(
        self,
        wee: np.array,
        wei: np.array,
        te: np.array,
        x: np.array,
        y: np.array,
        white_noise_e: np.array,
    ):
        """Activity of Excitatory neurons in the network

        Args:
            wee(array): Excitatory-Excitatory weight matrix

            wei(array): Inhibitory-Excitatory weight matrix

            te(array): Excitatory threshold

            x(array): Excitatory network activity

            y(array): Inhibitory network activity

            white_noise_e(array): Gaussian noise

        Returns:
            x(array): Current Excitatory network activity
        """
        xt = x[:, 1][:, None]
        yt = y[:, 1][:, None]

        incoming_drive_e = np.expand_dims(
            self.incoming_drive(weights=wee, activity_vector=xt), 1
        )
        incoming_drive_i = np.expand_dims(
            self.incoming_drive(weights=wei, activity_vector=yt), 1
        )
        tot_incoming_drive = (
            incoming_drive_e
            - incoming_drive_i
            + white_noise_e
            + np.asarray(self.v_t)
            - te
        )

        # Heaviside step function
        heaviside_step = np.expand_dims([0.0] * len(tot_incoming_drive), 1)
        heaviside_step[tot_incoming_drive > 0] = 1.0
        return heaviside_step

    def inhibitory_network_state(
        self, wie: np.array, ti: np.array, y: np.array, white_noise_i: np.array
    ):
        """Activity of Excitatory neurons in the network

        Args:
            wee(array): Excitatory-Excitatory weight matrix

            wie(array): Excitatory-Inhibitory weight matrix

            ti(array): Inhibitory threshold

            y(array): Inhibitory network activity

            white_noise_i(array): Gaussian noise

        Returns:
            y(array): Current Inhibitory network activity"""

        wie = np.asarray(wie)
        yt = y[:, 1][:, None]
        incoming_drive_e = np.expand_dims(
            self.incoming_drive(weights=wie, activity_vector=yt), 1
        )

        tot_incoming_drive = incoming_drive_e + white_noise_i - ti
        heaviside_step = np.expand_dims([0.0] * len(tot_incoming_drive), 1)
        heaviside_step[tot_incoming_drive > 0] = 1.0

        return heaviside_step

    def recurrent_drive(
        self,
        wee: np.array,
        wei: np.array,
        te: np.array,
        x: np.array,
        y: np.array,
        white_noise_e: np.array,
    ):
        """Network state due to recurrent drive received by the each unit at time t+1. Activity of Excitatory neurons without external stimuli

        Args:

            wee(array): Excitatory-Excitatory weight matrix

            wei(array): Inhibitory-Excitatory weight matrix

            te(array): Excitatory threshold

            x(array): Excitatory network activity

            y(array): Inhibitory network activity

            white_noise_e(array): Gaussian noise

        Returns:
            xt(array): Recurrent network state
        """
        xt = x[:, 1][:, None]
        yt = y[:, 1][:, None]

        incoming_drive_e = np.expand_dims(
            self.incoming_drive(weights=wee, activity_vector=xt), 1
        )

        incoming_drive_i = np.expand_dims(
            self.incoming_drive(weights=wei, activity_vector=yt), 1
        )

        tot_incoming_drive = incoming_drive_e - incoming_drive_i + white_noise_e - te

        heaviside_step = np.expand_dims([0.0] * len(tot_incoming_drive), 1)
        heaviside_step[tot_incoming_drive > 0] = 1.0

        return heaviside_step


# Simulate / Train SORN
class Simulator_(Sorn):

    """Simulate SORN using external input/noise using the fresh or pretrained state

    Args:
        inputs(np.array, optional): External stimuli. Defaults to None.

        phase(str, optional): Plasticity phase. Defaults to "plasticity".

        state(dict, optional): Network states, connections and threshold state. Defaults to None.

        timesteps(int, optional): Total number of time steps to simulate the network. Defaults to 1.

        noise(bool, optional): If True, noise will be added. Defaults to True.

    Returns:
        plastic_state(dict): Network states, connections and threshold state

        X_all(array): Excitatory network activity collected during entire simulation steps

        Y_all(array): Inhibitory network activity collected during entire simulation steps

        R_all(array): Recurrent network activity collected during entire simulation steps

        frac_pos_active_conn(list): Number of positive connection strengths in the network at each time step during simulation"""

    def __init__(self):
        super().__init__()

        self.avail_callbacks = {
            "ExcitatoryActivation": ExcitatoryActivation,
            "InhibitoryActivation": InhibitoryActivation,
            "RecurrentActivation": RecurrentActivation,
            "WEE": WEE,
            "WEI": WEI,
            "TE": TE,
            "TI": TI,
            "EIConnectionCounts": EIConnectionCounts,
            "EEConnectionCounts": EEConnectionCounts,
        }

    def update_callback_state(self, *args) -> None:
        if self.callbacks:
            (indices,) = np.array(self.callback_mask).nonzero()
            keys = [list(self.avail_callbacks.keys())[i] for i in indices]

            values = [args[idx] for idx in indices]
            for key, val in zip(keys, values):
                self.callback_state[key] = val

    def run(
        self,
        inputs: np.array = None,
        phase: str = "plasticity",
        state: dict = None,
        timesteps: int = None,
        noise: bool = True,
        freeze: list = None,
        neurogenesis: bool = True,
        num_new_neurons: int = None,
        neurogenesis_init_step: int = None,
        callbacks: list = [],
        **kwargs,
    ):
        """Simulation/Plasticity phase

        Args:
            inputs(np.array, optional): External stimuli. Defaults to None.

            phase(str, optional): Plasticity phase. Defaults to "plasticity"

            state(dict, optional): Network states, connections and threshold state. Defaults to None.

            timesteps(int, optional): Total number of time steps to simulate the network. Defaults to 1.

            noise(bool, optional): If True, noise will be added. Defaults to True.

            freeze(list, optional): List of synaptic plasticity mechanisms which will be turned off during simulation. Defaults to None.

            neurogenesis(bool, optional): If True, new neurons will be added based on certain conditions. Defaults to True

            num_new_neurons(int, optional): Number of new neurons to create at excitatory pool during neurogenesis

            neurogenesis_init_step(int, optional): Step to start neurogenesis
            callbacks(list, optional): Requested values from ["ExcitatoryActivation", "InhibitoryActivation",
                                                            "RecurrentActivation", "WEE", "WEI", "TE", "EEConnectionCounts"] collected and returned from the simulate sorn object.

        Returns:
            plastic_state(dict): Network states, connections and threshold state

            callback_values(dict): Requexted network parameters and activations"""

        assert (
            phase == "plasticity" or "training"
        ), "Phase can be either 'plasticity' or 'training'"

        self.timesteps = timesteps
        Sorn.timesteps = timesteps
        self.phase = phase
        self.state = state
        self.freeze = [] if freeze == None else freeze
        self.callbacks = callbacks
        self.exc_genesis = True
        self.inh_genesis = True
        self.num_new_neurons = num_new_neurons
        self.neurogenesis_init_step = neurogenesis_init_step
        self.ne_init = Sorn.ne

        kwargs_ = [
            "ne",
            "nu",
            "network_type_ee",
            "network_type_ei",
            "network_type_ie",
            "lambda_ee",
            "lambda_ei",
            "lambda_ie",
            "eta_stdp",
            "eta_inhib",
            "eta_ip",
            "te_max",
            "ti_max",
            "ti_min",
            "te_min",
            "mu_ip",
            "sigma_ip",
        ]
        for key, value in kwargs.items():
            if key in kwargs_:
                setattr(Sorn, key, value)
        Sorn.ni = int(0.2 * Sorn.ne)

        plasticity = Plasticity()
        # Initialize/Get the weight, threshold state and activity vectors
        matrix_collection = MatrixCollection(phase=self.phase, state=self.state)

        if self.callbacks:
            assert isinstance(self.callbacks, list), "Callbacks must be a list"
            assert all(isinstance(callback, str) for callback in self.callbacks)
            self.callback_mask = np.isin(
                list(self.avail_callbacks.keys()), self.callbacks
            ).astype(int)
            self.callback_state = dict.fromkeys(self.callbacks, None)

            # Requested values to be collected and returned
            self.dispatcher = Callbacks(
                self.timesteps, self.avail_callbacks, self.callbacks
            )

        if self.exc_genesis:
            assert self.num_new_neurons != None, "Number of neurons value missing"
            assert self.neurogenesis_init_step != None, "Neurogenesis step missing"
            assert self.timesteps > abs(
                self.num_new_neurons - self.neurogenesis_init_step
            ), "Neurogenesis steps is higher than simulation time steps"

            self.genesis_times = generate_exp(
                size=self.num_new_neurons,
                rmin=self.neurogenesis_init_step,
                rmax=self.timesteps,
            )
            print(len(np.unique(self.genesis_times)))
        # To get the last activation status of Exc and Inh neurons
        for i in tqdm(range(self.timesteps)):

            Wee, Wei, Wie = (
                matrix_collection.Wee,
                matrix_collection.Wei,
                matrix_collection.Wie,
            )
            Te, Ti = matrix_collection.Te, matrix_collection.Ti
            X, Y = matrix_collection.X, matrix_collection.Y

            network_state = NetworkState(inputs[:, i])

            # Buffers to get the resulting x and y vectors at the current time step and update the master matrix
            x_buffer, y_buffer = np.zeros((Wee[i].shape[0], 2)), np.zeros(
                (Wei[i].shape[0], 2)
            )

            # Fraction of active connections between E-E and E-I networks
            ei_conn = (Wei[i] > 0.0).sum()
            ee_conn = (Wee[i] > 0.0).sum()

            if noise:
                white_noise_e = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Wee[i].shape[0]
                )
                white_noise_i = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Wei[i].shape[0]
                )
            else:
                white_noise_e, white_noise_i = 0.0, 0.0

            # Recurrent drive
            r = network_state.recurrent_drive(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e
            )

            # Get excitatory states and inhibitory states given the weights and thresholds
            # x(t+1), y(t+1)
            excitatory_state_xt_buffer = network_state.excitatory_network_state(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e
            )
            inhibitory_state_yt_buffer = network_state.inhibitory_network_state(
                Wie[i], Ti[i], X[i], white_noise_i
            )

            # Update X and Y
            x_buffer[:, 0] = X[i][:, 1]  # xt -->(becomes) xt_1
            x_buffer[
                :, 1
            ] = excitatory_state_xt_buffer.T  # New_activation; x_buffer --> xt

            y_buffer[:, 0] = Y[i][:, 1]
            y_buffer[:, 1] = inhibitory_state_yt_buffer.T

            # Plasticity phase
            plasticity = Plasticity()

            # STDP
            if "stdp" not in self.freeze:
                Wee[i] = plasticity.stdp(Wee[i], x_buffer, cutoff_weights=(0.0, 1.0))

            # Intrinsic plasticity
            if "ip" not in self.freeze:
                Te[i] = plasticity.ip(Te[i], x_buffer)

            # Structural plasticity
            if "sp" not in self.freeze:
                Wee[i] = plasticity.structural_plasticity(Wee[i])

            # iSTDP
            if "istdp" not in self.freeze:
                Wei[i] = plasticity.istdp(
                    Wei[i], x_buffer, y_buffer, cutoff_weights=(0.0, 1.0)
                )

            # TODO: Test condition for neurogenesis
            if self.exc_genesis:
                # Check genesis time
                if i in self.genesis_times:
                    # Check for inhibitory neurogenesis
                    if (Sorn.ne > self.ne_init) and ((Sorn.ne + 1) % 5 == 0):
                        self.inh_genesis = True
                    else:
                        self.inh_genesis = False
                    neurogenesis = Neurogenesis()
                    (
                        x_buffer,
                        y_buffer,
                        Wee[i],
                        Wei[i],
                        Wie[i],
                        Te[i],
                        Ti[i],
                    ) = neurogenesis.step(
                        exc_pool=self.exc_genesis,
                        inh_pool=self.inh_genesis,
                        x_buffer=x_buffer,
                        y_buffer=y_buffer,
                        wee=Wee[i],
                        wei=Wei[i],
                        wie=Wie[i],
                        te=Te[i],
                        ti=Ti[i],
                    )

            # Synaptic scaling Wee
            if "ss" not in self.freeze:
                Wee[i] = plasticity.ss(Wee[i])
                Wei[i] = plasticity.ss(Wei[i])

            # Assign the state to the matrix collections
            matrix_collection.weight_matrix(Wee[i], Wei[i], Wie[i], i)
            matrix_collection.threshold_matrix(Te[i], Ti[i], i)
            matrix_collection.network_activity_t(x_buffer, y_buffer, i)
            if self.callbacks:
                self.update_callback_state(
                    x_buffer[:, 1],
                    y_buffer[:, 1],
                    r,
                    Wee[i],
                    Wei[i],
                    Te[i],
                    Ti[i],
                    ei_conn,
                    ee_conn,
                )

                self.dispatcher.step(self.callback_state, time_step=i)

        plastic_state = {
            "Wee": matrix_collection.Wee[-1],
            "Wei": matrix_collection.Wei[-1],
            "Wie": matrix_collection.Wie[-1],
            "Te": matrix_collection.Te[-1],
            "Ti": matrix_collection.Ti[-1],
            "X": X[-1],
            "Y": Y[-1],
        }
        print(matrix_collection.Wee[-1].shape)
        if self.callbacks:
            return plastic_state, self.dispatcher.get()
        else:
            return plastic_state, {}


class Trainer_(Sorn):
    """Train the network with the fresh or pretrained network state and external stimuli"""

    def __init__(self):
        super().__init__()
        self.avail_callbacks = {
            "ExcitatoryActivation": ExcitatoryActivation,
            "InhibitoryActivation": InhibitoryActivation,
            "RecurrentActivation": RecurrentActivation,
            "WEE": WEE,
            "WEI": WEI,
            "TE": TE,
            "TI": TI,
            "EIConnectionCounts": EIConnectionCounts,
            "EEConnectionCounts": EEConnectionCounts,
        }

    def update_callback_state(self, *args) -> None:
        if self.callbacks:
            (indices,) = np.array(self.callback_mask).nonzero()
            keys = [list(self.avail_callbacks.keys())[i] for i in indices]

            values = [args[idx] for idx in indices]
            for key, val in zip(keys, values):
                self.callback_state[key] = val

    def train_sorn(
        self,
        inputs: np.array = None,
        phase: str = "train",
        state=None,
        timesteps: int = None,
        noise: bool = True,
        freeze: list = None,
        callbacks: list = [],
        **kwargs,
    ):
        """Train the network with the fresh or pretrained network state and external stimuli

            Args:
            inputs(np.array, optional): External stimuli. Defaults to None.

            phase(str, optional): Training phase. Defaults to "training".

            state(dict, optional): Network states, connections and threshold state. Defaults to None.

            timesteps(int, optional): Total number of time steps to simulate the network. Defaults to 1.

            noise(bool, optional): If True, noise will be added. Defaults to True.

            freeze(list, optional): List of synaptic plasticity mechanisms which will be turned off during simulation. Defaults to None.

            max_workers(int, optional): Maximum workers for multhreading the plasticity steps

        Returns:
            plastic_state(dict): Network states, connections and threshold state

            X_all(array): Excitatory network activity collected during entire simulation steps

            Y_all(array): Inhibitory network activity collected during entire simulation steps

            R_all(array): Recurrent network activity collected during entire simulation steps

            frac_pos_active_conn(list): Number of positive connection strengths in the network at each time step during simulation"""

        assert (
            phase == "plasticity" or "training"
        ), "Phase can be either 'plasticity' or 'training'"

        kwargs_ = [
            "ne",
            "nu",
            "network_type_ee",
            "network_type_ei",
            "network_type_ie",
            "lambda_ee",
            "lambda_ei",
            "lambda_ie",
            "eta_stdp",
            "eta_inhib",
            "eta_ip",
            "te_max",
            "ti_max",
            "ti_min",
            "te_min",
            "mu_ip",
            "sigma_ip",
        ]
        for key, value in kwargs.items():
            if key in kwargs_:
                setattr(Sorn, key, value)
        Sorn.ni = int(0.2 * Sorn.ne)

        self.phase = phase
        self.state = state
        self.timesteps = timesteps
        Sorn.timesteps = timesteps
        self.inputs = np.asarray(inputs)
        self.freeze = [] if freeze == None else freeze
        self.callbacks = callbacks
        matrix_collection = MatrixCollection(phase=self.phase, state=self.state)

        if self.callbacks:
            assert isinstance(self.callbacks, list), "Callbacks must be a list"
            assert all(isinstance(callback, str) for callback in self.callbacks)
            self.callback_mask = np.isin(
                list(self.avail_callbacks.keys()), self.callbacks
            ).astype(int)
            self.callback_state = dict.fromkeys(self.callbacks, None)

            # Requested values to be collected and returned
            self.dispatcher = Callbacks(
                self.timesteps, self.avail_callbacks, self.callbacks
            )

        for i in range(self.timesteps):

            if noise:
                white_noise_e = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Sorn.ne
                )
                white_noise_i = Initializer.white_gaussian_noise(
                    mu=0.0, sigma=0.04, t=Sorn.ni
                )
            else:
                white_noise_e = 0.0
                white_noise_i = 0.0

            network_state = NetworkState(self.inputs[:, i])

            # Buffers to get the resulting x and y vectors at the current time step and update the master matrix
            x_buffer, y_buffer = np.zeros((Sorn.ne, 2)), np.zeros((Sorn.ni, 2))

            Wee, Wei, Wie = (
                matrix_collection.Wee,
                matrix_collection.Wei,
                matrix_collection.Wie,
            )
            Te, Ti = matrix_collection.Te, matrix_collection.Ti
            X, Y = matrix_collection.X, matrix_collection.Y

            # Fraction of active connections between E-E and E-I networks
            ei_conn = (Wei[i] > 0.0).sum()
            ee_conn = (Wee[i] > 0.0).sum()

            # Recurrent drive at t+1 used to predict the next external stimuli
            r = network_state.recurrent_drive(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e=white_noise_e
            )

            # Get excitatory states and inhibitory states given the weights and thresholds
            # x(t+1), y(t+1)
            excitatory_state_xt_buffer = network_state.excitatory_network_state(
                Wee[i], Wei[i], Te[i], X[i], Y[i], white_noise_e=white_noise_e
            )
            inhibitory_state_yt_buffer = network_state.inhibitory_network_state(
                Wie[i], Ti[i], X[i], white_noise_i=white_noise_i
            )

            # Update X and Y
            x_buffer[:, 0] = X[i][:, 1]  # xt -->xt_1
            x_buffer[:, 1] = excitatory_state_xt_buffer.T  # x_buffer --> xt
            y_buffer[:, 0] = Y[i][:, 1]
            y_buffer[:, 1] = inhibitory_state_yt_buffer.T

            if self.phase == "plasticity":
                # Plasticity phase
                plasticity = Plasticity()
                # STDP
                if "stdp" not in self.freeze:
                    Wee[i] = plasticity.stdp(
                        Wee[i], x_buffer, cutoff_weights=(0.0, 1.0)
                    )

                # Intrinsic plasticity
                if "ip" not in self.freeze:
                    Te[i] = plasticity.ip(Te[i], x_buffer)

                # Structural plasticity
                if "sp" not in self.freeze:
                    Wee[i] = plasticity.structural_plasticity(Wee[i])

                # iSTDP
                if "istdp" not in self.freeze:
                    Wei[i] = plasticity.istdp(
                        Wei[i], x_buffer, y_buffer, cutoff_weights=(0.0, 1.0)
                    )

                # Synaptic scaling Wee
                if "ss" not in self.freeze:
                    Wee[i] = plasticity.ss(Wee[i])
                    Wei[i] = plasticity.ss(Wei[i])

            else:
                # Wee[i], Wei[i], Te[i] remain same
                pass

            # Assign the state to the matrix collections
            matrix_collection.weight_matrix(Wee[i], Wei[i], Wie[i], i)
            matrix_collection.threshold_matrix(Te[i], Ti[i], i)
            matrix_collection.network_activity_t(x_buffer, y_buffer, i)
            if self.callbacks:
                self.update_callback_state(
                    x_buffer[:, 1],
                    y_buffer[:, 1],
                    r,
                    Wee[i],
                    Wei[i],
                    Te[i],
                    Ti[i],
                    ei_conn,
                    ee_conn,
                )

                self.dispatcher.step(self.callback_state, time_step=i)

        plastic_state = {
            "Wee": matrix_collection.Wee[-1],
            "Wei": matrix_collection.Wei[-1],
            "Wie": matrix_collection.Wie[-1],
            "Te": matrix_collection.Te[-1],
            "Ti": matrix_collection.Ti[-1],
            "X": X[-1],
            "Y": Y[-1],
        }

        if self.callbacks:
            return plastic_state, self.dispatcher.get()
        else:
            return plastic_state, {}


Trainer = Trainer_()
Simulator = Simulator_()
if __name__ == "__main__":
    pass
