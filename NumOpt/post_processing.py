import matplotlib.pyplot as plt
import numpy as np

def plot_histories(histories, labels, xlog=(False, False), ylog=(False, True)):
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left subplot: function value ---
    for i,hist in enumerate(histories):
        fval = np.array(hist['fval'])
        grad = np.array(hist['grad'])
        gradMag = np.zeros(fval.shape)
        for k in range(len(gradMag)):
            gradMag[k] = np.linalg.norm(grad[k])
        
        axes[0].plot(fval[1:], '-', mfc='none', label=labels[i])
        axes[1].plot(gradMag[1:], '-', mfc='none')
    
    axes[0].set_title("Function Value")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("f")
    axes[0].grid(alpha=0.2)

    # --- Right subplot: gradient magnitude ---
    axes[1].set_title("Gradient Magnitude")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("|∇f|")
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.2)
    
    axes[0].legend()

    if xlog[0]:
        axes[0].set_xscale('log')
    if ylog[0]:
        axes[0].set_yscale('log')

    if xlog[1]:
        axes[1].set_xscale('log')
    if ylog[1]:
        axes[1].set_yscale('log')


    plt.tight_layout()