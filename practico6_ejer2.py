#!/usr/bin/env python3
"""
Created on Sun Aug 25 20:41:09 2024

@author: luisvargas
"""

import matplotlib.pyplot as plt
import numpy as np

from ode import integrador_ode, rk4


def f(r, t, p):
    x = r[0]
    y = r[1]
    z = r[2]
    s = p[0]
    r = p[1]
    b = p[2]
    return np.array([s * (y - x), (r * x) - y - (x * z), (x * y) - (b * z)])


# 2.1.i)

s = 10.0  # parametros
r = 0.5
b = 8.0 / 3.0
p = np.array([s, r, b])
x0 = 1.0  # condiciones iniciales
y0 = 0.5
z0 = 0.1
r0 = np.array([x0, y0, z0])  # inicio de trayectoria


a = 0
b = 20
h = 0.01
k = int((b - a) / h)

t, r = integrador_ode(rk4, f, r0, a, b, k, p)
t1 = t
x1, y1, z1 = r


# 2.1.ii)

s = 10.0  # parametros
r = 10.0
b = 8.0 / 3.0
p = np.array([s, r, b])
x0 = 1.0  # condiciones iniciales
y0 = 0.5
z0 = 0.1
r0 = np.array([x0, y0, z0])  # inicio de trayectoria


a = 0
b = 10
h = 0.01
k = int((b - a) / h)

t, r = integrador_ode(rk4, f, r0, a, b, k, p)
t2 = t
x2, y2, z2 = r


# 2.1.iii)

s = 10.0  # parametros
r = 28.0
b = 8.0 / 3.0
p = np.array([s, r, b])
x0 = 1.0  # condiciones iniciales
y0 = 1.0
z0 = 1.0
r0 = np.array([x0, y0, z0])  # inicio de trayectoria


a = 0
b = 50
h = 0.01
k = int((b - a) / h)

t, r = integrador_ode(rk4, f, r0, a, b, k, p)
t3 = t
x3, y3, z3 = r


plt.xlabel("$t$ [ms]")
plt.ylabel("")


plt.plot(t1, x1, label="x(t)", linestyle="-", c="red")
plt.plot(t1, y1, label="y(t)", linestyle="-", c="blue")
plt.plot(t1, z1, label="z(t)", linestyle="-", c="green")
plt.show()

plt.plot(t2, x2, label="x(t)", linestyle="-", c="red")
plt.plot(t2, y2, label="y(t)", linestyle="-", c="blue")
plt.plot(t2, z2, label="z(t)", linestyle="-", c="green")
plt.show()

plt.plot(t3, x3, label="x(t)", linestyle="-", c="red")
plt.plot(t3, y3, label="y(t)", linestyle="-", c="blue")
plt.plot(t3, z3, label="z(t)", linestyle="-", c="green")
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x1, y1, z1)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x2, y2, z2)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x3, y3, z3)
plt.show()
