import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, gamma

# --- Configuration Parameters ---
MU = 10.0  # Mean E[X]
PHI = 0.2  # Dispersion Parameter φ
N_SAMPLES = 50000  # Number of samples for simulation
P_VALUES = [1.2, 1.4, 1.6, 1.8]  # Tweedie power parameter values

def tweedie_sampler_full(p, mu, phi, n_samples):
    """
    Samples from the Tweedie family covering p=1 (Poisson), 1 < p < 2 (Compound), and p=2 (Gamma).
    """
    if p == 1.0:
        # p=1.0: Poisson Distribution (Discrete count data)
        # Var(X) = mu (phi assumed to be 1 for standard Poisson)
        return poisson.rvs(mu, size=n_samples)
    
    elif p == 2.0:
        # p=2.0: Gamma Distribution (Continuous positive values)
        # Var(X) = phi * mu^2. Parameters derived from mean/variance relationship.
        alpha_k = 1 / phi
        beta_theta = mu * phi
        return gamma.rvs(a=alpha_k, scale=beta_theta, size=n_samples)
    
    elif 1.0 < p < 2.0:
        # 1 < p < 2: Compound Poisson-Gamma Distribution (Custom logic required)
        
        # 1. Calculate Poisson mean (lambda) for the number of events
        lambda_p = (mu**(2 - p)) / ((2 - p) * phi)
        
        # 2. Calculate Gamma shape (alpha_k) and scale (beta_theta) for event size
        alpha_k = (2 - p) / (p - 1)
        beta_theta = phi * (p - 1) * (mu**(p - 1))
        
        # 3. Perform Sampling
        n_events = poisson.rvs(lambda_p, size=n_samples) # Number of gamma events
        samples = np.zeros(n_samples)
        
        for i in range(n_samples):
            if n_events[i] > 0:
                # Summing n_events[i] independent Gamma random variables
                samples[i] = np.sum(gamma.rvs(a=alpha_k, scale=beta_theta, size=n_events[i]))
        return samples

    else:
        raise ValueError("Supported range for p is [1.0, 2.0].")

# --- Execute Plotting (All distributions on one plot) ---
fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, p in enumerate(P_VALUES):
    try:
        data = tweedie_sampler_full(p, MU, PHI, N_SAMPLES)

        ax.hist(data, bins=150, density=True, color=colors[i], alpha=0.5,
                edgecolor='none', linewidth=0, label=f'p={p:.1f}')

    except ValueError as e:
        print(f'Error for p={p}: {e}')

ax.set_title(f'Tweedie Distribution Shape Change (μ={MU}, $\\phi$={PHI})', fontsize=16)
ax.set_xlabel('X (Value)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tweedie_distribution.png', dpi=300, bbox_inches='tight')
print('Graph saved as tweedie_distribution.png')
plt.show()