import numpy as np


def entropy(x: np.array):
    """Shannon Entropy of a vector
    H(X) = -sum(P(X)log2(P(X)))

    Args:
        x (array): 1D Array or activations of a layer or input feature vector

    Returns:
        entropy (float): Shannon entropy of the vector x
    """
    _, count = np.unique(x, return_counts=True, axis=0)
    prob = count / len(x)  # propability of elements in x
    h_x = np.sum((-1) * prob * np.log2(prob))
    return h_x


def joint_entropy(x: np.array, y: np.array):
    """Compute Joint entropy between two vectors
    H(X,Y) = -sum(P(X,Y)log2(P(X,Y))
    Args:
        x (array): 1D Array or activations of a layer or feature vector 
        y (array): 1D Array or activations of another layer or feature vector

    Returns:
        float: Joint entropy of X and Y
    """
    # Note the dimensions of X and Y should be same
    xy = np.c_[x, y]  # [[x1,y1], [x2,y2]...[xn,yn]]
    h_xy = entropy(xy)
    return h_xy


def conditional_entropy(x: np.array, y: np.array):
    """Compute conditional entropy of X given Y
    H[X|Y] = H(X,Y) - H(Y)
    Args:
        x (array): 1D Array or activations of a layer or feature vector 
        y (array): 1D Array or activations of another layer or feature vector
    """
    c_x_y = joint_entropy(x, y) - entropy(y)
    return c_x_y


def mutual_information(x: np.array, y: np.array):

    """Compute Mutual information/Information gain between variables X and Y as
    I(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X)+ H(Y)-H(X,Y) 
    Args:
        x (array): 1D Array or activations of a layer or feature vector 
        y (array): 1D Array or activations of another layer or feature vector

    Returns:
        mi (float): Mutual information between X and Y  
    """
    I = entropy(x) - conditional_entropy(x, y)
    return I


def partial_information_decomposition(x: np.array, y: np.array):
    return NotImplementedError
