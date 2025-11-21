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


    def solve(self, x0=None, directionMethod='steepest_descent',
          stepMethod='backtracking', options=None):

        # safety
        if self.objFunction is None:
            raise ValueError("Objective function not set.")

        # options
        maxiter = options.get('maxiter', 500) if options else 500
        tol = options.get('tol', 1e-5) if options else 1e-5
        mu1 = options.get('mu1', 1e-4) if options else 1e-4
        mu2 = options.get('mu2', 0.9) if options else 0.9

        # initialization
        if x0 is None:
            x0 = np.random.rand(self.nDim)

        x = np.array(x0, dtype=float)
        fval = self.objFunction(x, *self.objFunctionArgs)
        grad = self.gradFunction(self.objFunction, x, *self.objFunctionArgs)

        # initial direction (STEEREST DESCENT = raw negative gradient)
        direction = -grad

        # store history
        self.history['x'].append(x.copy())
        self.history['fval'].append(fval)
        self.history['grad'].append(grad.copy())
        self.history['dir'].append(direction.copy())
        
        reset = False

        # iteration loop
        for k in range(maxiter):

            # ✔ correct convergence check
            if np.linalg.norm(grad) < tol:
                break

            # ---- compute search direction ----
            if directionMethod == 'steepest_descent':
                direction = -grad

            elif directionMethod == 'conjugate_gradient':

                if k == 0 or k % self.nDim == 0:   # hard reset
                    direction = -grad
                    beta = 0

                else:
                    gradOld = self.history['grad'][-2]
                    directionOld = self.history['dir'][-2]

                    # ----- Polak-Ribiere beta -----
                    yk = grad - gradOld
                    denom = np.dot(gradOld, gradOld)

                    if denom < 1e-20:
                        beta = 0.0
                    else:
                        beta_PR = np.dot(grad, yk) / denom
                        beta = max(beta_PR, 0.0)     # PR+ truncation

                    # cap very large values (numerical stability)
                    beta = min(beta, 1e3)

                    # ----- Powell orthogonality restart -----
                    if abs(np.dot(grad, gradOld)) / denom > 0.2:
                        beta = 0.0

                    # ----- Update direction -----
                    direction = -grad + beta * directionOld

                    # ----- Descent direction check -----
                    if np.dot(direction, grad) >= 0:
                        direction = -grad
                        beta = 0.0

            elif directionMethod == 'quasi_newton':
                if k == 0 or reset:
                    Vk = np.eye(self.nDim) / np.linalg.norm(grad)    # scaled identity inverse
                    reset = False

                else:
                    sk = self.history['x'][-1] - self.history['x'][-2]
                    yk = self.history['grad'][-1] - self.history['grad'][-2]

                    ys = yk @ sk

                    # curvature condition (this must hold for positive definiteness)
                    if ys > 1e-10:
                        rho = 1.0 / ys

                        I = np.eye(self.nDim)
                        outer_sy = np.outer(sk, yk)
                        outer_ys = np.outer(yk, sk)
                        outer_ss = np.outer(sk, sk)

                        Vk = (I - rho * outer_sy) @ Vk @ (I - rho * outer_ys) + rho * outer_ss

                        # enforce symmetry to avoid numerical issues that could break positive definiteness
                        Vk = 0.5 * (Vk + Vk.T)
                    else:
                        # restart if curvature condition fails
                        reset = True
                        Vk = np.eye(self.nDim) / np.linalg.norm(grad)

                direction = -Vk @ grad

                # ensure it's a descent direction for the linesearch method
                if grad @ direction >= 0:
                    Vk = np.eye(self.nDim) / np.linalg.norm(grad)
                    direction = -Vk @ grad
                    reset = True
                    
            else:
                raise NotImplementedError(f"Direction '{directionMethod}' not implemented.")

            # ---- line search ----
            if stepMethod == 'backtracking':
                alpha = self.linesearchBacktracking(x, direction, step=1.0, mu=mu1)

            elif stepMethod == 'strong_wolfe':
                alpha = self.lineseachBracketingPinPoint(x, direction, initialStep=1.0, c1=mu1, c2=mu2)

            elif stepMethod == 'fixed':
                alpha = options.get('step_size', 1e-2) if options else 1e-2

            else:
                raise NotImplementedError(f"Line search '{stepMethod}' not implemented.")

            # ---- update ----
            x = x + alpha * direction
            fval = self.objFunction(x, *self.objFunctionArgs)
            grad = self.gradFunction(self.objFunction, x, *self.objFunctionArgs)

            # store history
            self.history['x'].append(x.copy())
            self.history['dir'].append(direction.copy())
            self.history['fval'].append(fval)
            self.history['grad'].append(grad.copy())

        return self.history



    def plotSolutionHistory(self, history):
        """Plot trajectory of the optimization. Only for 2D problems

        Args:
            history (_type_): _description_
        """
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
        """Plot decrease of function value and gradient magnitude
        """
        
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
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.2)

        plt.tight_layout()
        

    def linesearchBacktracking(self, x0, dir, step=1.0, mu=1e-4, maxIter=100):
        """Backtracking algorithm for linesearch methods. 

        Args:
            x0 (np.array): current point
            dir (np.array): direction of linesearch
            step (float, optional): step magnitude. Defaults to 1.0, normalized.
            mu (float, optional): acceptance criterion of function decrease. Defaults to 1e-4.
            maxIter (float, optional): max iter of backtracking alg. Defaults to 100.

        Returns:
            step (float): entity of the step along search direction
        """
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


    def lineseachBracketingPinPoint(self, x0, d, initialStep, c1=1e-4, c2=0.5, rho=2.0):
        """Bracketing+Pinpoint alg. for linesearch methods. Satisfying strong wolfe conditions

        Args:
            x0 (np.array): current point
            d (np.array): direction of linesearch
            initialStep (float, optional): step magnitude. Defaults to 1.0, normalized.
            c1 (float, optional): acceptance criterion of function decrease. Defaults to 1e-4.
            c2 (float, optional): acceptance criterion of curvature. Defaults to 0.5.
            rho (float, optional): step increase of alg. if bracketing fails.

        Returns:
            step (float): entity of the step along search direction
        """
        # Initial values needed for conditions
        f0 = self.objFunction(x0, *self.objFunctionArgs)
        grad0 = self.gradFunction(self.objFunction, x0, *self.objFunctionArgs)
        df0 = np.dot(grad0, d)

        if df0 >= 0:
            raise ValueError("Direction is not a descent direction")

        alpha1 = 0.0
        alpha2 = initialStep
        
        f1 = f0
        df1 = df0

        first = True
        while True:

            f2 = self.objFunction(x0 + alpha2 * d, *self.objFunctionArgs)
            grad2 = self.gradFunction(self.objFunction, x0 + alpha2 * d, *self.objFunctionArgs)
            df2 = np.dot(grad2, d)

            # Armijo OR f2 > f1 → bracket found
            if (f2 > f0 + c1 * alpha2 * df0) or (not first and f2 > f1):
                return self.pinpoint(x0, d, alpha1, alpha2, f1, f0, df0, c1, c2)

            # Strong Wolfe curvature
            if np.abs(df2) <= -c2 * df0:
                return alpha2

            # Derivative sign switch → bracket found
            if df2 >= 0:
                return self.pinpoint(x0, d, alpha2, alpha1, f2, f0, df0, c1, c2)

            # Otherwise expand upwards
            alpha1 = alpha2
            f1 = f2
            df1 = df2
            alpha2 = rho * alpha2
            first = False

            

    def pinpoint(self, x0, d, alpha_lo, alpha_hi, phi_lo, phi0, dphi0, c1, c2, maxIter=20):
        """
        Find the step size for the given direction d using the pin-point
        method, which is a variant of the bisection method.

        Parameters
        ----------
        x0 : array
            Initial point
        d : array
            Search direction
        alpha_lo : float
            Lower bound of the step size
        alpha_hi : float
            Upper bound of the step size
        phi_lo : float
            Value of the objective function at x0 + alpha_lo * d
        phi0 : float
            Value of the objective function at x0
        dphi0 : float
            Derivative of the objective function at x0 in the direction d
        c1 : float
            Parameter for the Armijo condition
        c2 : float
            Parameter for the Strong Wolfe condition
        maxIter : int, optional
            Maximum number of iterations (default is 20)

        Returns
        -------
        alpha : float
            Step size
        """
        for i in range(maxIter):

            # Bisection point (can be improved with cubic interpolation)
            alpha = 0.5 * (alpha_lo + alpha_hi)

            x = x0 + alpha * d
            phi = self.objFunction(x, *self.objFunctionArgs)
            g = self.gradFunction(self.objFunction, x, *self.objFunctionArgs)
            dphi = np.dot(g, d)

            x_lo = x0 + alpha_lo * d
            phi_lo_val = self.objFunction(x_lo, *self.objFunctionArgs)

            # Condition 1 — Armijo violation OR phi >= phi_lo
            if (phi > phi0 + c1 * alpha * dphi0) or (phi >= phi_lo_val):
                alpha_hi = alpha
            else:
                # Condition 2 — Strong curvature satisfied
                if abs(dphi) <= -c2 * dphi0:
                    return alpha

                # Derivative sign switch
                if dphi * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo

                alpha_lo = alpha

        return alpha  # fallback
    
    