import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
x=np.linspace(0,2*np.pi,100)
y=np.sin(x) + 0.1*np.random.randn(100)

result=lowess(y,x,frac=0.3)


plt.scatter(x,y,alpha=0.7)
plt.plot(result[:,0],result[:,1],linewidth=2)
plt.title("locally weightes regression")
plt.legend(), plt.grid(alpha=0.3)
plt.show()