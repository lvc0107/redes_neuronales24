#!/usr/bin/env python3
"""
Created on Sun Aug 25 20:41:09 2024

@author: luisvargas
"""

import numpy as np

from ode import integrador_ode, rk4

s = 10
r = 0.5
b = 8.0 / 3.0
p = (s, r, b)


a = 0
b = 10
h = 0.5
k = int((b - a) / h)
xa = 0


def f(r, t, p):
    x = r[0]
    y = r[1]
    z = r[2]
    s = p[0]
    r = p[1]
    b = p[2]
    return np.array([s * (y - x)], (r * x) - y - (x * z), (x * y) - (b * z))


x0 = 1.0
y0 = 0.5
z0 = 0.1
r0 = np.array([x0, y0, z0])


t, w = integrador_ode(rk4, f, r0, a, b, k, p)
print(w[0, :])
