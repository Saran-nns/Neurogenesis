import unittest
import pickle
import numpy as np
from sorn.sorn import Trainer, Simulator
from sorn.utils import Plotter, Statistics

# Getting back the pickled matrices:
with open("sample_matrices.pkl", "rb") as f:
    (
        state_dict,
        Exc_activity,
        Inh_activity,
        Rec_activity,
        num_active_connections,
    ) = pickle.load(f)

# Default Test inputs
simulation_inputs = np.random.rand(10, 2)
gym_input = np.random.rand(10, 1)
sequence_input = np.random.rand(10, 1)

# Overriding defaults: Sample input
num_features = 10
timesteps = 1000
inputs = np.random.rand(num_features, timesteps)


class TestNeurogenesis(unittest.TestCase):
    def test_neurogenesis(self):
        # Initialize and simulate SORN with the default hyperparameters
        self.assertRaises(
            Exception,
            Simulator.simulate_sorn(
                inputs=simulation_inputs,
                phase="plasticity",
                matrices=None,
                timesteps=2,
                noise=True,
                nu=num_features,
                neurogenesis=True,
                neurogenesis_init_step=10,
                num_new_neurons=100,
            ),
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
