from scipy import log
from scipy.special import zeta
from scipy.optimize import bisect
import numpy as np
from scipy.special import zeta
from scipy import sqrt
from complexnetwork.avalanche import Avalanches


class PowerLaw(object):

    """Recover alpha from the discrete power law obeying data using Riemann Zeta function
    """

    def __init__(self):
        pass

    def exponent(self, x: np.array, xmin: int, a: float = 1.01, b: float = 10.0):
        """Compute alpha of the power law distribution. Basic bisection routine to find a zero of the 
        function f between the arguments a and b. f(a) and f(b) cannot have the same signs

        Args:
            x (np.array): Data points, could be avalanche durations, size, iai.
            xmin (int): The point from which you are interested to observe the power law behavior in the discrete data. 
            a (float, optional): One end of the bracketing interval [a,b]. Defaults to 1.01.
            b (float, optional): The other end of the bracketing interval [a,b]. Defaults to 10.0.
        Returns:
            alpha (float): Power law exponent
        """

        def log_zeta(x):
            return log(zeta(x, 1))

        def log_dzeta(x):
            h = 1e-5
            return (log_zeta(x + h) - log_zeta(x - h)) / (2 * h)

        t = -sum(log(x / xmin)) / len(x)

        def objective(x):
            return log_dzeta(x) - t

        return bisect(objective, a, b, xtol=1e-6)

    def confidence_interval(self, n: int, alpha_hat: float):

        """Compute the confidence interval sigma confidence interval to give an idea how much uncertainty 
           remains in power law exponent estimate
        
        Args:
            n (float): The alpha value at which we assume the avalanche behavoir kicks in
            
            alpha_hat (float): Estimated power law exponent
        
        Returns:
            sigma (float): Confidence value of power law exponent estimate
        """

        def zeta_prime(x, xmin=1):
            h = 1e-5
            return (zeta(x + h, xmin) - zeta(x - h, xmin)) / (2 * h)

        def zeta_double_prime(x, xmin=1):
            h = 1e-5
            return (zeta(x + h, xmin) - 2 * zeta(x, xmin) + zeta(x - h, xmin)) / h ** 2

        def sigma(n, alpha_hat, xmin=1):
            z = zeta(alpha_hat, xmin)
            temp = zeta_double_prime(alpha_hat, xmin) / z
            temp -= (zeta_prime(alpha_hat, xmin) / z) ** 2
            return 1 / sqrt(n * temp)

        return sigma(n, alpha_hat)


def test_powerlaw(X: np.array, activity_threshold: float = 1.0):

    (_, _, avalanche_sizes, _,) = Avalanches().avalanche_observables(
        X, activity_threshold=activity_threshold
    )

    avalanche_size_hist = Avalanches.size_histogram(avalanche_sizes)

    xmin = 1
    # Avalanche histogram
    x = avalanche_size_hist[1]
    a, b = 1.01, 10
    # Exponent estimate
    alpha_hat = PowerLaw().exponent(x, xmin, a, b)
    # Confidence interval of the above estimate, assuming the network shows at n, the avalanche behavoir kicks in
    conf_alpha = PowerLaw().confidence_interval(n=1.0, alpha_hat=alpha_hat)
