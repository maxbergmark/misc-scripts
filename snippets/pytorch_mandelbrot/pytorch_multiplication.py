import torch
from torchvision.utils import save_image

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def iterate(b_c, x_min, x_max, x_dim, y_min, y_max, y_dim, iters):
	n_points = 10**7
	s_p = torch.rand(n_points, 2)
	s_p[:,0] *= y_max - y_min + 2
	s_p[:,1] *= x_max - x_min + 2
	s_p[:,0] += y_min - 1
	s_p[:,1] += x_min - 1
	i_p = s_p.clone()
	for i in range(iters):
		# print("first point", i_p[0,:])

		# temp = i_p[:,0]
		# i_p[:,0] = i_p[:,0]*i_p[:,0] - i_p[:,1]*i_p[:,1] + s_p[:,0]
		# i_p[:,1] = 2*temp*i_p[:,1] + s_p[:,1]

		i_p[:,0], i_p[:,1] = (
			i_p[:,0]*i_p[:,0] - i_p[:,1]*i_p[:,1] + s_p[:,0],
			2*i_p[:,0]*i_p[:,1] + s_p[:,1]
		)

	dists = (i_p*i_p).sum(1)
	dists[dists != dists] = 1000
	# print(dists)
	s2 = s_p[dists > 3, :]
	if s2.size() == torch.Size([0]):
		return
	s2_r = s2[:,0]
	s2_c = s2[:,1]
	i2 = s2.clone()
	# i2_r = i2[:,0]
	# i2_c = i2[:,1]

	for i in range(iters):

		# temp = i2[:,0].clone()
		# i2[:,0] = i2[:,0]*i2[:,0] - i2[:,1]*i2[:,1] + s2_r
		# i2[:,1] = 2*temp*i2[:,1] + s2_c
		i2[:,0], i2[:,1] = (
			i2[:,0]*i2[:,0] - i2[:,1]*i2[:,1] + s2[:,0],
			2*i2[:,0]*i2[:,1] + s2[:,1]
		)

		pixels = i2.clone()
		pixels[:,0] -= y_min
		pixels[:,1] -= x_min
		pixels[:,0] /= y_max - y_min
		pixels[:,1] /= x_max - x_min
		pixels[:,0] *= y_dim
		pixels[:,1] *= x_dim
		pixels = pixels.long()
		p_f = pixels[
			(pixels[:,0] >= 0)
			& (pixels[:,1] >= 0)
			& (pixels[:,0] < y_dim)
			& (pixels[:,1] < x_dim)
		]
		# print(i2[0,:], i2.size())
		if (p_f.size() != torch.Size([0])):
			b_c[p_f[:,0], p_f[:,1]] += 1
		else:
			break

def make_mandelbrot(x_min, x_max, x_dim, y_min, y_max, y_dim):

	b_c = torch.zeros(y_dim, x_dim).float()
	iters = 20
	generations = 10
	for i in range(generations):
		print("\r\titeration: %d/%d" % (i, generations), end='', flush=True)
		iterate(b_c, x_min, x_max, x_dim, y_min, y_max, y_dim, iters)

	s = b_c.sum()
	b_c /= b_c.max()
	b_c = torch.min(1.5*b_c, b_c*.1+.9)
	# b_c = b_c**5
	# b_c[:,0] = 1
	# b_c[0,:] = 1
	# b_c[:,x_dim-1] = 1
	# b_c[y_dim-1,:] = 1
	save_image(b_c, "buddha_%dx%d_%d_%d.png" % (x_dim, y_dim, iters, s))

y_min, y_max = -3.2, 1.5
x_min, x_max = -1.5, 1.5
x_dim, y_dim = 1440, 2560

make_mandelbrot(x_min, x_max, x_dim, y_min, y_max, y_dim)