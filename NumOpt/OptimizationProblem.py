import numpy as np
from NumOpt.gradient import complexStepGradient
from NumOpt.linesearch import linesearchBacktracking
import matplotlib.pyplot as plt
# from NumOpt.styles import *

class OptimizationProblem:
    def __init__(self, nDim, bounds=None):
        """
        Initialize the optimization problem.
        
        Parameters:
        - dim: int, number of decision variables
        - bounds: list of tuples, (min, max) for each variable
        """
        self.nDim = nDim
        self.bounds = bounds
        self.solution = None
        self.objValue = None
        self.gradFunction = complexStepGradient
        self.minHistory = []


    def setObjectiveFunction(self, x, *fargs):
        """
        Define the objective function.
        Override this method in subclass or instance.
        """
        self.objFunction = x
        self.objFunctionArgs = fargs


    def setConstraints(self, x):
        """
        Define constraints.
        Should return a list of dicts as required by scipy.optimize.
        Example: [{'type': 'ineq', 'fun': lambda x: x[0] - 1}]
        """
        raise NotImplementedError("setConstraints method not implemented.")
        return []
    
    
    def setGradientMethod(self, method='complex_step'):
        if method == 'complex_step':
            self.gradFunction = complexStepGradient
        else:
            raise NotImplementedError(f"Gradient method '{method}' not implemented.")


    def solve(self, x0=None, directionMethod='steepest_descent', stepMethod='backtracking', options=None):
        
        
        #secutiry check
        if self.objFunction is None:
            raise ValueError("Objective function not set.")
        
        # initial guess
        if x0 is None:
            x0 = np.random.rand(self.nDim)
        x = np.array(x0, dtype=float)
        self.minHistory.append(x.copy())
        
        # convergence criteria
        maxiter = options.get('maxiter', 1000) if options else 1000
        tol = options.get('tol', 1e-6) if options else 1e-6

        # start iterations
        for k in range(maxiter):
            fval = self.objFunction(x, *self.objFunctionArgs)
            grad = self.gradFunction(self.objFunction,x, *self.objFunctionArgs)
            
            if np.linalg.norm(grad) < tol:
                break

            if directionMethod == 'steepest_descent':
                direction = - grad/np.linalg.norm(grad)
            else:
                raise NotImplementedError(f"Direction '{direction}' not implemented.")

            if stepMethod == 'backtracking':
                alpha = linesearchBacktracking(self.objFunction, x, direction, self.gradFunction)
            else:
                raise NotImplementedError(f"Line search '{stepMethod}' not implemented.")  # fixed small step if no line search

            x = x + alpha * direction
            self.minHistory.append(x.copy())

        self.solution = x
        self.objValue = self.objFunction(x)
        return self.minHistory


    def plotSolutionHistory(self, history):
        
        assert self.nDim == 2, "Plotting only implemented for 2D problems."
        N = 250
        
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], N)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        F = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                F[i, j] = self.objFunction([X[i, j], Y[i, j]], *self.objFunctionArgs)
        
        plt.figure()
        plt.contourf(X, Y, F, levels=25, cmap='viridis')
        plt.contour(X, Y, F, levels=25, colors='k', linewidths=0.25)
        
        history = np.array(history)
        plt.plot(history[:, 0], history[:, 1], 'r--^', label='Trajectory')
        plt.plot(history[0, 0], history[0, 1], 'k^', label='Start')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Optimization trajectory - Iterations: {}'.format(len(history)-1))
        plt.legend()
        




