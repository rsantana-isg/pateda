import numpy as np

def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2)."""
    return np.sum(x**2)


def ellipsoidal_function(x, alpha=1e6):
    """
    Computes the value of the ellipsoidal function.

    Args:
        x (numpy.ndarray): Input vector of shape (n,), where n is the dimensionality.
        alpha (float): Scaling factor for the quadratic terms. Default is 1e6.

    Returns:
        float: The value of the ellipsoidal function at x.
    """
    n = len(x)
    return sum([alpha**(i/(n - 1)) * (x[i] ** 2) for i in range(n)])



def schaffers_f7_function(x):
    """
    Computes the value of the Schaffers F7 function.

    Args:
        x (numpy.ndarray): Input vector of shape (n,), where n is the dimensionality.

    Returns:
        float: The value of the Schaffers F7 function at x.
    """
    n = len(x)
    s = np.sqrt(x[:-1]**2 + x[1:]**2)  # Compute s_i = sqrt(x_i^2 + x_{i+1}^2)
    term1 = np.sqrt(s)  # First term: sqrt(s_i)
    term2 = np.sqrt(s) * np.sin(50 * s**0.2)**2  # Second term: sqrt(s_i) * sin^2(50 * s_i^0.2)
    sum_terms = np.sum(term1 + term2)  # Sum over all i
    return (sum_terms / (n - 1))**2  # Square the average

