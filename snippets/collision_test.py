import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


def draw_circles(p, r):
	n = p.shape[0]
	for i in range(n):
		t = np.linspace(0, 2*np.pi)
		x = p[i, 0] + r[i] * np.cos(t)
		y = p[i, 1] + r[i] * np.sin(t)
		plt.plot(x, y)

def iterate(p, v, r):
	dt = 0.2
	n = p.shape[0]
	colls = defaultdict(list)
	for i in range(n):
		p0 = p[i,:]
		v0 = v[i,:]
		r0 = r[i]
		for j in range(i+1, n):
			p1 = p[j,:]
			v1 = v[j,:]
			r1 = r[j]
			p_dist = np.linalg.norm(p0-p1)
			v_dist = np.linalg.norm(v0-v1)

			p_dist = np.linalg.norm(p0-p1)
			v_dist = np.linalg.norm(v0-v1)
			qp = 2*np.dot(p0-p1, v0-v1) / v_dist**2
			qq = (p_dist**2 - (r0+r1)**2) / v_dist**2
			if qp*qp/4 > qq:
				t = -qp/2 - (qp*qp/4-qq)**.5
				if 0 < t < dt:
					colls[i].append((j, t))
					colls[j].append((i, t))

	for i in range(n):
		if len(colls[i]) == 0:
			p[i,:] += v[i,:] * dt
		else:
			j, t = colls[i][0]
			p[i,:] += v[i,:] * t


	for i in range(n):
		if len(colls[i]) > 0:
			j, t = colls[i][0]
			p_rel = p[i,:] - p[j,:]
			# print(p_rel)
			n = p_rel / np.linalg.norm(p_rel)
			v[i,:] -= 2*np.dot(v[i,:], n) * n
			p[i,:] += v[i,:] * (dt-t)


n = 10
# p = np.array([[np.random.random(), np.random.random()] ])
p = np.random.rand(n, 2)*100
# v = np.array([[3.0, 0.0], [0.0, 2.0]])
v = np.random.randn(n, 2)*10
r = np.abs(np.random.randn(n))
draw_circles(p, r)

for i in range(200):
	iterate(p, v, r)
	draw_circles(p, r)
plt.axis('square')


plt.xlim([0, 100])
plt.ylim([0, 100])
plt.show()