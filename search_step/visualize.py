import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.0, 10000.0, 1.0)
fig, ax = plt.subplots(1, 5)

a = 0.01
b = 1000
c = 0.55

y = a * (1 + x / b) ** (-c)
ax[0].plot(x, y)

a = 100 ** 3
b = 0.001
c = 0.33

y = (a + x / b) ** (-c)
ax[1].plot(x, y)

a = 100 ** 3
b = 0.000001
c = 0.33

y = (a + x / b) ** (-c)
ax[2].plot(x, y)


y = x ** (-1/3.0)
ax[3].plot(x, y)


ax[4].plot([x for x in xrange(0, 11)], [0.001*(10**(-0.3))**x for x in xrange(0, 11)])


plt.show()