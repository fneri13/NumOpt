import numpy as np
from NumOpt.gradient import complexStepGradient

def backtracking(f, x0, *fargs, maxIter=100, tol=1e-6, alpha0=1.0, mu=1e-4):
    """backtracking algorithm for step decision

    Args:
        f (_type_): function
        x0 (_type_): initial x
        maxIter (int, optional): max number of x updates. Defaults to 100.
        tol (_type_, optional): value of grad norm to stop. Defaults to 1e-6.
        alpha0 (float, optional): initial step magnitude. Defaults to 1.0.
        mu (_type_, optional): acceptance criterion parameter. Defaults to 1e-4.

    Returns:
        _type_: min and history of locations
    """
    k = 0
    fval = f(x0, *fargs)
    grad = complexStepGradient(f, x0, *fargs)
    alpha = alpha0
    xhist = [x0.copy()]
    
    while (np.linalg.norm(grad) > tol and k < maxIter):
        gradDir = grad / np.linalg.norm(grad)
        xnew = x0 - alpha * gradDir
        fnew = f(xnew, *fargs)
        if fnew < fval - mu * alpha * np.dot(grad, gradDir):
            x0 = xnew
            fval = fnew
            grad = complexStepGradient(f, x0, *fargs)
            xhist.append(xnew)
            k += 1
        else:
            alpha *= 0.5
            
    return x0, xhist