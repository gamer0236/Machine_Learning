import matplotlib.pyplot as plt
import numpy as np

plt.ion()
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.show()
