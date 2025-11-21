def rosenbrock(x, a=1, b=100):
    """Two-dimensional rosenbrock function
    """
    x1, x2 = x
    return (a - x1)**2 + b*(x2 - x1**2)**2

def rosenbrock_ndim(x, a=1, b=100):
    """N_dimensional rosenbrock function
    """
    ndim = len(x)
    f = 0
    for i in range(0, ndim-1):
        f += b * (x[i+1] - x[i]**2)**2 + (a - x[i])**2
    return f

def himmelblau(x):
    """Two-dimensional himmelblau function
    """
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2


def beanFunction(x):
    """Two-dimensional bean function
    """
    x1, x2 = x
    f = (1 - x1)**2 + (1-x2)**2 + 0.5 *(2*x2-x1**2)**2
    return f


