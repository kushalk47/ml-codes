import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Generate synthetic data (sinusoidal with noise)
X = np.linspace(0, 2 * np.pi, 100)
y = np.sin(X) + 0.1 * np.random.randn(100)

# Apply LOWESS (Locally Weighted Regression)
result = lowess(y, X, frac=0.3)  # frac = smoothing parameter (window size)

# Plot
plt.scatter(X, y, label='Data', color='red', alpha=0.7)
plt.plot(result[:, 0], result[:, 1], label='LOWESS Fit', color='blue', linewidth=2)
plt.xlabel('X'), plt.ylabel('y'), plt.title('Locally Weighted Regression')
plt.legend(), plt.grid(alpha=0.3)
plt.show()
