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
        self.history = {"x": [], "fval": [], "grad": [], "dir": []}


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
        
        # security check
        if self.objFunction is None:
            raise ValueError("Objective function not set.")
        
        # convergence criteria
        maxiter = options.get('maxiter', 1000) if options else 1000
        tol = options.get('tol', 1e-6) if options else 1e-6
        
        # initialization
        if x0 is None:
            x0 = np.random.rand(self.nDim)
        x = np.array(x0, dtype=float)
        fval = self.objFunction(x, *self.objFunctionArgs)
        grad = self.gradFunction(self.objFunction,x, *self.objFunctionArgs)
        direction = -grad/np.linalg.norm(grad)
        self.history['x'].append(x)
        self.history['fval'].append(fval)
        self.history['grad'].append(grad)
        self.history['dir'].append(direction)
        
        # start iterations
        for k in range(maxiter):
            
            # check convergence
            if np.linalg.norm(grad) < tol:
                break
            
            # compute search direction
            if directionMethod == 'steepest_descent':
                direction = - grad/np.linalg.norm(grad)
            elif (directionMethod == 'conjugate_gradient'):
                if k == 0 or k%(self.nDim*1) == 0: 
                    direction = - grad/np.linalg.norm(grad)
                else:
                    gradOld = self.history['grad'][-2]
                    directionOld = self.history['dir'][-2]
                    # beta = np.dot(grad, (grad - gradOld)) / np.dot(gradOld, gradOld)
                    # beta = max(0, beta)  # ensure beta is non-negative
                    beta = np.dot(grad, grad) / np.dot(gradOld, gradOld)
                    direction = - grad + beta * directionOld
                    direction = direction / np.linalg.norm(direction)
            else:
                raise NotImplementedError(f"Direction '{direction}' not implemented.")

            # compute step size
            if stepMethod == 'backtracking':
                alpha = linesearchBacktracking(self.objFunction, x, direction, self.gradFunction)
            elif stepMethod == 'fixed':
                alpha = options.get('step_size', 1e-2) if options else 1e-2
            else:
                raise NotImplementedError(f"Line search '{stepMethod}' not implemented.")  # fixed small step if no line search

            # update solution
            x = x + alpha * direction
            fval = self.objFunction(x, *self.objFunctionArgs)
            grad = self.gradFunction(self.objFunction,x, *self.objFunctionArgs)
            self.history['x'].append(x)
            self.history['dir'].append(direction)
            self.history['fval'].append(fval)
            self.history['grad'].append(grad)

        return self.history


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
        
        x = np.array(history['x'])
        plt.plot(x[:, 0], x[:, 1], 'r--^', label='Trajectory')
        plt.plot(x[0, 0], x[0, 1], 'k^', label='Start')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title('Optimization trajectory - Iterations: {}'.format(len(x)-1))
        plt.legend()
        
        
        plt.figure()
        fval = np.array(history['fval'])
        plt.semilogy(fval)
        plt.xlabel('Iteration')
        plt.ylabel('Objective function value')
        plt.grid(alpha=0.2)
        




