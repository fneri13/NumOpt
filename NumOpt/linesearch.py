import numpy as np
from NumOpt.gradient import complexStepGradient

def linesearchBacktracking(func, x0, dir, gradFunc, *fargs, step0=1.0, mu=1e-4, maxIter=100):
    """_summary_

    Args:
        func (_type_): _description_
        x0 (_type_): _description_
        dir (_type_): _description_
        gradFunc (_type_): _description_
        step0 (float, optional): _description_. Defaults to 1.0.
        mu (_type_, optional): _description_. Defaults to 1e-4.
        maxIter (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    gradOld = gradFunc(func, x0, *fargs)
    fOld = func(x0, *fargs)
    
    # initialize iterations
    x = x0 + step0 * dir
    fVal = func(x, *fargs)
    step = step0
    it = 0
    
    # start iterations
    while (fVal > fOld - mu * np.dot(gradOld, step*dir)) and it < maxIter:
        gradOld = gradFunc(func, x, *fargs)
        step *= 0.5
        x = x0 + step * dir
        fVal = func(x, *fargs)
        it += 1
        
    return step