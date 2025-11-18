import numpy as np
from skopt.learning.gaussian_process.kernels import Matern, RBF, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct
from skopt.learning import GaussianProcessRegressor
from scipy.optimize import minimize
from scipy.optimize import basinhopping, differential_evolution


def lower_confidence_bound(X, model, lambda_):
    """
    Lower Confidence Bound acquisition function.
    
    LCB(x) = mu(x) - lambda * sigma(x)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)  # Ensure shape is (1, n_features)
    mu, sigma = model.predict(X, return_std=True)
    return mu - lambda_ * sigma

def optimize_acquisition_direct(model, bounds, lambda_, n_candidates=10):
    def acq(x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mu, sigma = model.predict(x, return_std=True)
        return (mu - lambda_ * sigma)[0]

    candidates = []
    for _ in range(n_candidates):
        result = differential_evolution(acq, bounds, maxiter=1)
        #print(result.success,result.x, result.fun)
        #if result.success:
        candidates.append((result.x, result.fun))

    # Sort and select N best
    candidates.sort(key=lambda x: x[1])
    return np.array([x for x, _ in candidates[:n_candidates]])

def optimize_using_acquisition_function(model, bounds, lambda_, N, n_candidates=10):
    def acq(x):
        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        mu, sigma = model.predict(x, return_std=True)
        return (mu - lambda_ * sigma)[0]

    candidates = []
    for _ in range(n_candidates):
        x0 = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds]
        )
        result =  acq(x0)   #basinhopping(acq, x0,niter=10)
        #if result.success:
        candidates.append((x0, result))

    # Sort and select N best
    candidates.sort(key=lambda x: x[1])
    return np.array([x for x, _ in candidates[:N]])


def optimize_acquisition(model, bounds, lambda_, n_starts=20):
    """Optimizes the LCB acquisition function using L-BFGS-B from multiple restarts."""
    n_features = len(bounds)
    candidates = []

    for _ in range(n_starts):
        x0 = np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds]
        )

        res = minimize(
            fun=lower_confidence_bound,
            x0=x0,
            args=(model, lambda_),
            method="direct", #"L-BFGS-B",  #"COBYLA", #,
            bounds=bounds,
            options={'maxiter':10}
        )
        #print(x0, res.success, res.x, res.fun)
        #if res.success:
        candidates.append((res.x, res.fun))

    # Sort by acquisition function value (lower is better)
    candidates.sort(key=lambda x: x[1])
    return np.array([x for x, _ in candidates])
        
def BO_initialize_population(M, y, lower_bound, upper_bound, N, n_candidates, lambda_, ker):
    """
    Sample N candidate vectors from the GP that are likely to have low objective function values
    based on the Lower Confidence Bound (LCB) acquisition function.

    Args:
        M (np.ndarray): Input matrix of shape (n_samples, n_features).
        y (np.ndarray): Objective function values of shape (n_samples,).
        N (int): Number of likely optimal vectors to return.
        n_candidates (int): Number of candidate points to sample from the input space.
        lambda_ (float): Exploration-exploitation tradeoff parameter for LCB.

    Returns:
        X_selected (np.ndarray): Matrix of N selected vectors (N x n_features).
        mu (np.ndarray): Predicted mean at selected points.
        sigma (np.ndarray): Predicted std deviation at selected points.
    """
    n_samples, n_features = M.shape    
    bounds = [(lower_bound[i],upper_bound[i]) for i in range(n_features)]

    # Fit GP model

    if ker==0:
        kernel = Matern(length_scale_bounds=(1e-8, 1e5))
    elif ker==1:
        kernel =  RBF(length_scale=1.0, length_scale_bounds=(1e-8, 1e5))
    elif ker==2:
        kernel =  RationalQuadratic(length_scale=1.0, alpha=0.1)
    elif ker==3:
        kernel = ConstantKernel(0.1, (1e-8, 1e5))  * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)
        
    model = GaussianProcessRegressor(kernel=kernel,
                                     normalize_y=True)
  
    model.fit(M, y)

    #candidate_scaled = optimize_acquisition(model, bounds, lambda_, n_starts=n_candidates)
    #X_selected = optimize_acquisition_direct(model, bounds, lambda_, n_candidates)
    X_selected = optimize_using_acquisition_function(model, bounds, lambda_, n_candidates, N)
     

    return X_selected


def initialize_population(population_size, n, lower_bound, upper_bound):
    """
    Initialize a population of random solutions within the given bounds.
    
    :param population_size: Number of individuals in the population.
    :param n: Number of dimensions (variables) in each solution.
    :param lower_bound: Lower bound of the search space.
    :param upper_bound: Upper bound of the search space.
    :return: A numpy array of shape (population_size, n) containing the initial population.
    """

    X = np.random.uniform(lower_bound, upper_bound, (population_size, n))
    
    return X

