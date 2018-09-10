import numpy as np
from scipy.interpolate import interp2d
from scipy.integrate import quad
from scipy.misc import derivative

def partial_derivative(func, var=0, point=[]):
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)

    return derivative(wraps, point[var], dx=1e-6)


def poin_int(arr_x, arr_y, alpha=5, res=50, r=1, x0=0, y0=0):
    xvec = np.linspace(-alpha, alpha, res)
    X, Y = np.meshgrid(xvec, xvec)  # create grid
    x_func = interp2d(X, Y, arr_x, kind='cubic')  # interpolate data
    y_func = interp2d(X, Y, arr_y, kind='cubic')  # interpolate data

    def _x(t):  # x-coordinate of circular contour
        return np.cos(t) * r + x0

    def _dx(t):  # derivate of _x(t)
        return -np.sin(t) * r

    def _y(t):  # y-coordinate of circular contour
        return np.sin(t) * r + y0

    def _dy(t):  # derivative of _y(t)
        return np.cos(t) * r

    def _integrand1(t):  # integrand of first integral
        return (x_func(_x(t), _y(t)) *
                (partial_derivative(y_func, 0, [_x(t), _y(t)]) * _dx(t) +
                 partial_derivative(y_func, 1, [_x(t), _y(t)]) * _dy(t)) /
                (x_func(_x(t), _y(t))**2 + y_func(_x(t), _y(t))**2))

    def _integrand2(t):  # integrand of second integral
        return (-y_func(_x(t), _y(t)) *
                (partial_derivative(x_func, 0, [_x(t), _y(t)]) * _dx(t) +
                 partial_derivative(x_func, 1, [_x(t), _y(t)]) * _dy(t)) /
                (x_func(_x(t), _y(t))**2 + y_func(_x(t), _y(t))**2))

    _int1 = quad(_integrand1, 0, 2*np.pi, epsabs=1e-2)  # should be an integer
    _int2 = quad(_integrand2, 0, 2*np.pi, epsabs=1e-2)  # should be an integer
    return _int1, _int2

n = 10
arr_x = np.array([[i+j for j in range(n)]for i in range(n)])
arr_y = np.array([[i+j+1 for j in range(n)]for i in range(n)])
poin_int(arr_x, arr_y, 5, n)