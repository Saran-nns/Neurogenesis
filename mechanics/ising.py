import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class IsingModel(object):

    """Ising model of the reservoir pool acitivity at various time scales"""

    def __init__(self):
        pass

    def running_mean(self, X: np.array, window_size: int):

        """Measure the moving average of neural firing rate given the time window

        Args:
            X (np.array): Neural activity
            
            window_size (int): Time window size

        Returns:
            list: Moving average with the size equals len(X)//window_size
        """

        cumsum = np.cumsum(np.insert(X, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    def compute_spin(self, X: np.array, window_sizes: list = [1]):

        """Compute Sigma {-1,1}^N of neural population activity vector, given window sizes for the mean

        Args:
            X (np.array): Neural activity
            
            window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].

        Returns:
            dict : Neural spins at different time scales
        """

        # Placeholder for sigma of neural population
        sigma = {}
        sigma.fromkeys(window_sizes)
        for window in window_sizes:
            sigma_ = []
            for neuron in list(range(X.shape[1])):
                mean = self.running_mean(X[:, neuron], window)
                mean[mean > 0] = 1
                mean[mean <= 0] = -1
                sigma_.append(list(mean))
            sigma["%s" % window] = np.asarray(
                sigma_
            ).transpose()  # Rows= time dim, col= neurons
        return sigma

    def mean_spiking_activity(self, X: np.array, window_sizes: list = [1]):

        """Compute mean spiking probability of each neuron given their spins at different time scales

        Args:
            X (np.array): Neural activity
            
            window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].

        Returns:
            dict : Mean spiking probability of each neuron given their spins at different time scales
        """

        # Compute spins with respect to window size
        sigma = self.compute_spin(X, window_sizes=window_sizes)
        # Compute mean of each neuron's spin variance
        m = {}
        m.fromkeys(window_sizes)
        for window in window_sizes:
            m["%s" % window] = np.mean(sigma["%s" % window], axis=1)
        return m

    def compute_Q(self, X: np.array, window_sizes: list = [1]):

        """Compute two point function between pairs of neurons in the network

        Args:
            X (np.array): Neural activity
            
            window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].

        Returns:
            Q (dict): Two point function between pairs of neurons in the network
            
            sigma (dict): Neural spins at different time scales
        """
        num_neurons = X.shape[1]
        # Compute two-point function of spins with respect to window size
        sigma = self.compute_spin(X, window_sizes=window_sizes)

        # Two point function between neurons given their spin
        Q = {}
        Q.fromkeys(window_sizes)
        q = np.zeros((num_neurons, num_neurons))
        for window in window_sizes:
            for i in list(range(1, num_neurons)):
                for j in range(i):
                    q[i][j] = np.mean(
                        sigma["%s" % window][:, i] * sigma["%s" % window][:, j]
                    )
            q_ = q + q.T  # Is symmetric
            np.fill_diagonal(q_, 1)
            Q["%s" % window] = q_
        return Q, sigma

    def plotQ(self, Q: dict):

        """Plot Q, the two point function values between pairs of neurons in the network
        
        Args:
            Q (dict): Two point function between pairs of neurons in the network
        
        Returns:
            plot
        """

        # Create subplots
        ax_size = len(Q.keys())
        assert ax_size != 0, "Missing time window"

        if ax_size != 1:
            # Check is ax_size even or odd
            if ax_size % 2 != 0:
                ax_size += 1
            rows, cols = ax_size // 2, ax_size // 2

            fig, ax = plt.subplots(nrows=rows, ncols=cols, sharey=True)
            fig.suptitle("Q at various time scales")
            Q_vals = list(Q.values())
            Q_keys = list(Q.keys())
            for i in range(rows):
                for j in range(cols):

                    im = ax[i, j].imshow(Q_vals[i + j])

                    ax[i, j].legend([im], ["Timescale" + str(Q_keys[i])])

            cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.8])
            fig.colorbar(im, cax=cbar_ax)
            fig.legend()
            fig.tight_layout()

        else:

            plt.imshow(Q.values()[0])
            plt.title("Q at time scale" + str(Q.keys()[0]))
            plt.tight_layout()
            plt.colorbar(fraction=0.046, pad=0.04)

        return plt.show()


def test_ising(X, window_sizes=[1, 2, 5, 10, 20]):
    """Test avalanche module

    Args:
        X (np.array): Neural activity

        window_sizes (list, optional): Time window sizes to compute the spin at various time scales. Defaults to [1].
    """
    sigma = IsingModel().compute_spin(X, window_sizes=window_sizes)
    m = IsingModel().mean_spiking_activity(X, window_sizes=window_sizes)
    Q, sigma = IsingModel().compute_Q(X, window_sizes=window_sizes)
    p = IsingModel().plotQ(Q)


# if __name__ == "__main__":
#     import pickle

#     # Getting back the pickled matrices:
#     with open("../sample_matrices.pkl", "rb") as f:
#         (
#             matrices_dict,
#             Exc_activity,
#             Inh_activity,
#             Rec_activity,
#             num_active_connections,
#         ) = pickle.load(f)

#     X = np.asarray(Exc_activity)
#     activity_threshold = 1
#     window_sizes = [2, 5]
#     test_ising(X)
