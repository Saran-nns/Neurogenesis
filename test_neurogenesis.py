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

# Overriding defaults: Sample input
num_features = 10
timesteps = 1000
inputs = np.random.rand(num_features, timesteps)


class TestNeurogenesis(unittest.TestCase):
    def test_neurogenesis(self):
        # Initialize and simulate SORN with the default hyperparameters
        self.assertRaises(
            Exception,
            Simulator.run(
                ne=50,
                lamda_ei=5,
                inputs=inputs,
                phase="plasticity",
                matrices=None,
                timesteps=timesteps,
                noise=True,
                nu=num_features,
                neurogenesis=True,
                neurogenesis_init_step=100,
                num_new_neurons=15,
            ),
        )

# Run 

# def train():
    
#     num_features = 10
#     timesteps = 2000000
#     inputs = np.random.rand(num_features, timesteps)

#     state_dict, sim_dict, genesis_times = Simulator.run(
#                                         ne=50,
#                                         lamda_ei=5,
#                                         inputs=inputs,
#                                         phase="plasticity",
#                                         matrices=None,
#                                         timesteps=timesteps,
#                                         noise=True,
#                                         nu=num_features,
#                                         neurogenesis=True,
#                                         neurogenesis_init_step=timesteps//10,
#                                         num_new_neurons=150)
    
#     # Save the results
#     with open('sim_dict1.pickle', 'wb') as handle:
#         pickle.dump(sim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     with open('sim_dict1.pickle', 'rb') as handle:
#         sim_dict1 = pickle.load(handle)

#     with open('state_dict1.pickle', 'wb') as handle:
#         pickle.dump(state_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     with open('state_dict1.pickle', 'rb') as handle:
#         state_dict1 = pickle.load(handle)
        
#     with open('genesis_times.pickle', 'wb') as handle:
#         pickle.dump(genesis_times, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     with open('genesis_times.pickle', 'rb') as handle:
#         genesis_times = pickle.load(handle)

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
