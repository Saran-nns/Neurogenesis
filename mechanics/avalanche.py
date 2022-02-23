from matplotlib.pyplot import plot
import numpy as np
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter
from complexnetwork.utils import compute_spike_count


class Avalanches(object):
    def __init__(self):
        pass

    def avalanche_observables(self, X: np.array, activity_threshold: int = 1):

        """Avalanche sizes, durations and interval sizes
        
            - Set the neural activity =0 if < activity_threshold % of size of the network
            - Slice the array by non-zero value indices
            - Count the number of items in each slices: Duration of avalanches
            - Sum the items in each slices: Size of avalanches
            - Slice the array into zero value indices
            - Count number of items in each slices: Duration of inter avalanche intervals
        
        Args:
            X (np.array): Neural activity
            
            activity_threshold (int, optional): Threshold of number of spikes at each time step. Spike counts below threshold will be set to zero.Defaults to 1.

        Returns:
            spike_count (np.array): Number of spikes at each time step
            
            avalanche_durations (np.array): Avalanches durations
            
            avalanche_sizes (np.array): Number of spikes within each avalanche duration
            
            iai (np.array): Time interval between avalanches
        """

        spike_count = np.asarray(compute_spike_count(X))
        threshold = activity_threshold * X.shape[1] / 100
        spike_count[spike_count < threshold] = 0

        # Avalanche size and duration
        # Get the non zero indices
        aval_idx = np.nonzero(spike_count)[0]

        # Group indices by a consecutiveness
        aval_indices = []
        for k, g in itertools.groupby(enumerate(aval_idx), lambda ix: ix[0] - ix[1]):
            aval_indices.append(list(map(itemgetter(1), g)))

        # Using group indices, pick the correpondning items in the spike_count list
        avalanches = []
        for val in aval_indices:
            avalanches.append(list(spike_count[val]))

        # Avalanche sizes
        avalanche_sizes = [sum(avalanche) for avalanche in avalanches]
        # Avalanche duration
        avalanche_durations = [len(avalanche) for avalanche in avalanches]

        # Inter avalanche intervals

        # Get the indices where spike count =0
        silent_idx = np.where(spike_count == 0)[0]

        silent_indices = []
        # Group indices by consecutiveness
        for k, g in itertools.groupby(enumerate(silent_idx), lambda ix: ix[0] - ix[1]):
            silent_indices.append(list(map(itemgetter(1), g)))
        iai_ = []
        for val in silent_indices:
            iai_.append(list(spike_count[val]))
        # Duration of inter-avalanche intervals
        iai = [len(intervals) for intervals in iai_]

        return spike_count, avalanche_durations, avalanche_sizes, iai

    @staticmethod
    def avalanche_plot(avalanches: list):
        """Plot the avalanches observed

        Args:
            avalanches (list): Observed avalanches in the neural activity

        Returns:
            plot
        """

        plt.plot(avalanches, label="avalanche")
        plt.title("Avalanches")
        plt.tight_layout()
        plt.legend()
        plt.show()

    @staticmethod
    def plot(avalanches: list):
        """Plot the avalanches observed

        Args:
            avalanches (list): Observed avalanches in the neural activity

        Returns:
            plot
        """

        plt.plot(avalanches, label="avalanche")
        plt.title("Avalanches")
        plt.tight_layout()
        plt.legend()
        plt.show()

    @staticmethod
    def durations_histogram(durations: list, bin_size: int = 10, plot: bool = True):
        """Plot the avalanche durations observed in linear and log scales

        Args:
            durations (list): Observed avalanche durations in the neural activity
            
            bin_size (int): Number of bins for avalanche durations

            plot (bool): If True, plot the histogram in linear and logscale
            
        Returns:
            dur_hist (tuple): Histogram of obsered durations
        """

        # Histgram of avalanche sizes
        dur_hist = np.histogram(durations, bin_size)
        if plot:
            plt.figure(figsize=(16, 5))
            plt.subplot(221)
            plt.plot(dur_hist[1][:-1], dur_hist[0])
            plt.title("Avalanche durations in histogram in linear scale")
            plt.subplot(222)
            plt.plot(np.log(dur_hist[1][:-1]), np.log(dur_hist[0]))
            plt.title("Avalanche durations histogram in loglog scale")
            plt.show()
        return dur_hist

    @staticmethod
    def size_histogram(sizes: list, bin_size: int = 100, plot: bool = True):
        """ Histogram of avalanche size observed in linear and log scales 

        Args:
            durations (list): Observed avalanche durations in the neural activity
            
            bin_size (int): Number of bins for avalanche sizes
            
            plot (bool): If True, plot the histogram in linear and logscale

        Returns:
            size_hist (tuple): Histogram of obsered durations
        """

        # Histgram of avalanche sizes
        size_hist = np.histogram(sizes, bin_size)
        if plot:
            plt.figure(figsize=(16, 5))
            plt.subplot(221)
            plt.plot(np.log(size_hist[1][:-1]), np.log(size_hist[0]))
            plt.title("Avalanche sizes histogram in loglog scale")
            plt.subplot(222)
            plt.plot(size_hist[1][:-1], size_hist[0])
            plt.title("Avalanche sizes in histogram in linear scale")
            plt.show()

        return size_hist

    @staticmethod
    def iai_histogram(iai: list, bin_size: int = 100, plot: bool = True):
        """Plot the histogram of inter-avalanche intervals observed in linear and log scales 

        Args:
            iai (list): Observed inter-avalanche intervals in the neural activity
            
            bin_size (int): Number of bins for inter-avalanche intervals
            
            plot (bool): If True, plot the histogram in linear and logscale

        Returns:
            iai_hist: Histogram of obsered durations
        """

        # Histgram of avalanche sizes
        iai_hist = np.histogram(iai, bin_size)
        if plot:
            plt.figure(figsize=(16, 5))
            plt.subplot(221)
            plt.plot(np.log(iai_hist[1][:-1]), np.log(iai_hist[0]))
            plt.title("Avalanche sizes histogram in loglog scale")
            plt.subplot(222)
            plt.plot(iai_hist[1][:-1], iai_hist[0])
            plt.title("Avalanche sizes in histogram in linear scale")
            plt.show()
        return iai_hist


def test_avalanche(X: np.array, activity_threshold: int = 1):

    (
        avalanches,
        avalanche_durations,
        avalanche_sizes,
        iai,
    ) = Avalanches().avalanche_observables(X, activity_threshold=activity_threshold)

    Avalanches().plot(avalanches)
    Avalanches().durations_histogram(avalanche_durations, plot=True)
    Avalanches().size_histogram(avalanche_sizes, plot=True)
    Avalanches().iai_histogram(iai, plot=True)

