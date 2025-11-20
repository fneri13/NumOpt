import numpy as np
from NumOpt.gradient import complexStepGradient
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
                if k == 0 or k%self.nDim == 0: 
                    direction = - grad/np.linalg.norm(grad)
                else:
                    gradOld = self.history['grad'][-2]
                    directionOld = self.history['dir'][-2]
                    beta = np.dot(grad, grad) / np.dot(gradOld, gradOld)
                    direction = - grad + beta * directionOld
                    # direction = direction / np.linalg.norm(direction)
            else:
                raise NotImplementedError(f"Direction '{directionMethod}' not implemented.")

            # compute step size
            if stepMethod == 'backtracking':
                alpha = self.linesearchBacktracking(x, direction)
            elif stepMethod == 'fixed':
                alpha = options.get('step_size', 1e-2) if options else 1e-2
            else:
                raise NotImplementedError(f"Line search '{stepMethod}' not implemented.")  # fixed small step if no line search

            # update solution
            x = x + alpha * direction
            fval = self.objFunction(x, *self.objFunctionArgs)
            grad = self.gradFunction(self.objFunction,x, *self.objFunctionArgs)
            self.history['x'].append(x.copy())
            self.history['dir'].append(direction.copy())
            self.history['fval'].append(fval.copy())
            self.history['grad'].append(grad.copy())

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
        plt.gca().set_aspect('equal', adjustable='box')
        
    
    def plotFunctionDecreaseHistory(self, history):
        
        fval = np.array(history['fval'])
        grad = np.array(history['grad'])
        gradMag = np.zeros(fval.shape)
        for i in range(len(gradMag)):
            gradMag[i] = np.linalg.norm(grad[i])
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # --- Left subplot: function value ---
        axes[0].plot(fval, '-C0o')
        axes[0].set_title("Function Value")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("f")
        axes[0].grid(alpha=0.2)

        # --- Right subplot: gradient magnitude ---
        axes[1].plot(gradMag, '-C1s')
        axes[1].set_title("Gradient Magnitude")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("|∇f|")
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        

    def linesearchBacktracking(self, x0, dir, step=1.0, mu=1e-4, maxIter=100):

        # Reference values (always used in Armijo)
        f0 = self.objFunction(x0, *self.objFunctionArgs)
        g0 = self.gradFunction(self.objFunction, x0, *self.objFunctionArgs)

        it = 0
        x = x0 + step * dir
        fVal = self.objFunction(x, *self.objFunctionArgs)

        # Armijo condition: f(x0 + α d) <= f(x0) + μ α g0^T d
        while (fVal > f0 + mu * step * np.dot(g0, dir)) and it < maxIter:
            step *= 0.5
            x = x0 + step * dir
            fVal = self.objFunction(x, *self.objFunctionArgs)
            it += 1

        return step


