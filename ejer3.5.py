#!/usr/bin/env python3
"""
Created on Sun Aug 25 20:41:09 2024

@author: luisvargas
"""

import matplotlib as plt
import numpy as np

from ode import integrador_ode, rk4


def f(x, t, p):
    V = x[0]
    E = p[0]
    tau = p[1]
    R = p[2]
    I = p[3]

    return np.array([(E + R * I(t) - V) / tau])


I0 = 2


def I(t):
    return I0


def c(x, t, p):
    V = x[0]
    E = p[0]
    Vu = p[4]
    if V > Vu:
        V = E
    return np.array([V])


E = -65  # mV  reposo
a = 0
b = 100
h = 0.5
k = int((b - a) / h)
xa = np.array([E])
tau = 10
V0 = 10  # como usar?
R = 10
Vu = -50  # umbral
p = [E, tau, R, I, Vu]


# con disparo
t, w = integrador_ode(rk4, f, xa, a, b, k, p, c)
print(w[0, :])


# sin disparo
t2, w2 = integrador_ode(rk4, f, xa, a, b, k, p)
print(w2[0, :])


plt.xlabel("$t$ [ms]")
plt.ylabel("$V(t)$ [mV]")

valores_v = w[0, :]
valores_v2 = w2[0, :]


plt.plot(valores_t, valores_v, label="con disparo", linestyle="-", c="red")
