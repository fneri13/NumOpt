def rosenbrock(x, a=1, b=100):
    x1, x2 = x
    return (a - x1)**2 + b*(x2 - x1**2)**2

def himmelblau(x):
    x1, x2 = x
    return (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2