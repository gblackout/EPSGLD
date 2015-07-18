import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.0, 2000.0, 1.0)
fig, ax = plt.subplots(1, 6)

a = 1000000
b = 0.00001
c = 0.33

y = (a + x / b) ** (-c)
ax[0].plot(x, y)



a = 10000000
b = 0.00001
c = 0.33

y = (a + x / b) ** (-c)
ax[1].plot(x, y)



a = 100000
b = 0.00001
c = 0.33

y = (a + x / b) ** (-c)
ax[2].plot(x, y)


a = 1000000
b = 0.0001
c = 0.33

y = (a + x / b) ** (-c)
ax[3].plot(x, y)


a = 1000000
b = 0.000001
c = 0.33

y = (a + x / b) ** (-c)
ax[4].plot(x, y)




a = 0.01
b = 0.0001
c = 0.01
y = a * (1 + x / b) ** (-c)
ax[5].plot(x, y)

plt.show()