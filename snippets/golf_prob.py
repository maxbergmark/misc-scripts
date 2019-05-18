import numpy as np
from matplotlib import pyplot as plt

# generated timestamps for simulation
x = np.arange(0, 86400, 600)
# the function describing tee time demand
f = lambda x: np.sin(x*np.pi/3600/24)**2 * 0.6 + .0
prob = float(input("Desired booking density: "))

# expected value of f(x)
f_avg = f(x).sum() / x.size
# weighted function to make E(g) = sqrt(prob)
g = lambda x: prob + (1-prob) * (f(x) / f_avg / 2)

# empirically determined, needed to account for floor rounding in sampling
if prob < 1:
	correction = 1/(0.3935/prob**.5 + 0.8282*prob - 0.408*prob**2 + 0.2364)
else:
	correction = .99

# exponential factor to make E(X^gamma) = sqrt(prob)
gamma = (1-correction*prob**.5) / (correction*prob**.5) if prob else 1e9
# sample which tee times should have bookings
sample = (np.random.rand(x.size) < g(x)).astype(np.int32)
rands = np.random.rand(x.size)
# sample how many bookings each of those times should have
slots = (5*(rands**gamma)).astype(np.int32)
# apply bookings
sample *= slots

print(
	sample.sum(), 
	sample.size, 
	prob, 
	sample.sum() / sample.size / 4, 
	(sample > 0).sum()
)

# bookings
plt.plot(x, sample, '*')
# original probability function
plt.plot(x, f(x))
# weighted probability function
plt.plot(x, g(x))
# show the plot
plt.ylim([-.1, 5.5])
plt.show()