import numpy as np

def compute_spike_count(spike_train):    
    spike_count = np.count_nonzero(spike_train,1)
    # Filter zero entries in firing rate list above
    spike_count = list(filter(lambda a: a != 0, spike_count))
    return spike_count