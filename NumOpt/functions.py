import numpy as np

def complexStepGradient(f, x, h=1e-12):
    """
    Compute the gradient of f at point x using the complex step method.
    
    Parameters:
    f : callable
        The objective function.
    x : ndarray
        The point at which to compute the gradient.
    h : float
        The small step size for the complex step.
        
    Returns:
    grad : ndarray
        The gradient vector at point x.
    """
    n = len(x)
    grad = np.zeros(n, dtype=float)
    for i in range(n):
        x_step = np.array(x, dtype=complex)
        x_step[i] += 1j * h
        grad[i] = np.imag(f(x_step)) / h
    return grad