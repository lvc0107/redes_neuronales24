#!/usr/bin/env python3
"""
Created on Sun Aug 25 20:41:09 2024

@author: luisvargas
"""

import matplotlib.pyplot as plt
import numpy as np

from ode import integrador_ode, rk4


def alpha_n(v):
    return 0.01 * (10.0 - v) / (np.exp((10.0 - v) / 10.0) - 1.0)


def alpha_m(v):
    return 0.1 * (25.0 - v) / (np.exp((25.0 - v) / 10.0) - 1.0)


def alpha_h(v):
    return 0.07 * np.exp(-v / 20.0)


def beta_n(v):
    return 0.125 * np.exp(-v / 80.0)


def beta_m(v):
    return 4.0 * np.exp(-v / 18.0)


def beta_h(v):
    return 1.0 / (np.exp((30.0 - v) / 10.0) + 1.0)


def ina(x, p):
    """
    Ganancia de la corriente i por una canal de Sodio Na

    Parameters
    ----------
    x : TYPE
        DESCRIPTION. parametros para ?
    p : TYPE
        DESCRIPTION. parametros para?

    Returns
    -------
    La ganancia de iones que pasan a traves de un canal de sodio

     El Canal de Sodio tiene 3 compuertas de tipo m que se abren y dejan pasar
     la corriente y una de tipo h que debe estar inactiva para que dejen pasar
     la corriente
    """
    v = x[0]  # potencial
    m = x[2]  # probabilidad de que m este abierta
    h = x[3]  # probabilidad de que h este inactiva
    gna = p[1]  # gna
    vna = p[4]  # vna
    """
    para que pase i deben habrirse las 3 compuertas m y la h debe estar
    inactiva
    """
    pna = m**3 * h
    """
    gna * (v- vna) : conductividad promedio o ganancia promedio
    """
    return gna * (v - vna) * pna


def ik(x, p):
    """

    Ganancia de la corriente i por una canal de Potasio K

    Parameters
    ----------
    x : TYPE
        DESCRIPTION. parametros para probabilidades
    p : TYPE
        DESCRIPTION. parametros para formulas

    Returns
    -------
    La ganancia de iones que pasan a traves de un canal de Potasio

     El Canal de Sodio tiene 4 compuertas de tipo m que se abren y dejan
     pasar la corriente

    """
    v = x[0]
    n = x[1]
    gk = p[2]
    vk = p[5]
    """
    para que pase i deben habrirse las 4 compuertas de tipo n al mismo tiempo
    """
    pk = n**4
    return gk * (v - vk) * pk


def il(x, p):
    """

    perdida de la corriente i

    Parameters
    ----------
    x : TYPE
        DESCRIPTION. parametros para probabilidades
    p : TYPE
        DESCRIPTION. parametros para formulas


    Returns
    -------
        TODO completar

    """
    v = x[0]
    gl = p[3]
    vl = p[6]
    return gl * (v - vl)


# correspondientes tiempos caracteristicos y valores de equilibrios


def tau_n(v):
    return 1.0 / (alpha_n(v) + beta_n(v))


def tau_m(v):
    return 1.0 / (alpha_m(v) + beta_m(v))


def tau_h(v):
    return 1.0 / (alpha_h(v) + beta_h(v))


def n_inf(v):
    return alpha_n(v) * tau_n(v)


def m_inf(v):
    return alpha_m(v) * tau_m(v)


def h_inf(v):
    return alpha_h(v) * tau_h(v)


def f(x, t, p):
    """

    x[0] = v(t): potencial de la membrana
    x[1] = n(t): probabilidad de activacion de K (potasio)
    x[2] = m(t): probabilidad de activacion Na (sodio)
    x[3] = n(t): probabilidad de inactivacion Na (sodio)

    t: 𝑡∼5𝑚𝑠  : tiempo  en milisegundos

    p[0] = c = 1𝜇𝐹/𝑐𝑚2  : capacitancia de membrana (microF/cm **2)
    p[1] = 𝑔¯Na = 120𝑚𝑆/𝑐𝑚2: conductancia máxima de Na
    p[2] = 𝑔¯K = 36𝑚𝑆/𝑐𝑚2: conductancia máxima de  K
    p[3] = 𝑔𝑙 = 0.3𝑚𝑆/𝑐𝑚2: conductancia máxima de perdida
    p[4] = 𝑣Na=120𝑚𝑉: potencial de reversión de la corriente de Na
    (potencial de Nernst de Na)
    p[5] = 𝑣K=−12𝑚𝑉: potencial de reversión de la corriente de  K
    (potencial de Nernst de K)
    p[6] = 𝑣𝑙=10.6𝑚𝑉: potencial de reversión de la corriente de perdida
    (potencial de Nernst de la correinte de perdida)
    p[7] = 𝑖(𝑡)∼10𝜇𝐴/𝑐𝑚2: corriente de entrada al tiempo 𝑡

    """

    v = x[0]
    n = x[1]
    m = x[2]
    h = x[3]
    c = p[0]
    i = p[7]

    """
    Sistema de 4 ecuaciones ODE acopladas
    """
    return np.array(
        [
            # ecuacion para dv/dt
            (i(t) - ina(x, p) - ik(x, p) - il(x, p)) / c,
            alpha_n(v) * (1.0 - n) - beta_n(v) * n,  # ecuacion para dn/dt
            alpha_m(v) * (1.0 - m) - beta_m(v) * m,  # ecuacion para dm/dt
            alpha_h(v) * (1.0 - h) - beta_h(v) * h,  # ecuacion para dh/dt
        ]
    )


# EJERCICiO 2.2

# graficamos valores de equilibrio de las distintas fracciones de compuertas

plt.xlabel("$v$ [mV]")
plt.xlim(-50, 120)
plt.ylim(-0.1, 1.1)
v = np.linspace(-50, 120, 1000)

plt.plot(v, 0 * v, label="", linestyle="--", c="red")
plt.plot(v, 0 * v + 1, label="", linestyle="--", c="red")
plt.plot(v, np.vectorize(n_inf)(v), label="$n_{\\infty}$", c="orange")
plt.plot(v, np.vectorize(m_inf)(v), label="$m_{\\infty}$", c="green")
plt.plot(v, np.vectorize(h_inf)(v), label="$h_{\\infty}$", c="cyan")
plt.title("Valores de equilibrio de las distintas fracciones de compuertas abiertas")
plt.legend()
plt.show()

"""
EJERCICiO 2.3
Grafique los tiempos característicos de activación  𝜏𝑚 ,  𝜏𝑛  e
inactivación  𝜏ℎ
asociados a los distintos tipos de canales, en función de la diferencia de
potencial de membrana  𝑣 .
"""
plt.xlabel("$v$ [mV]")
plt.ylabel("$\\tau$ [ms]")

plt.xlim(-50, 120)
v = np.linspace(-50, 120, 1000)

plt.plot(v, 0 * v, label="", linestyle="--", c="blue")
plt.plot(v, np.vectorize(tau_n)(v), label="$\\tau_n$", c="orange")
plt.plot(v, np.vectorize(tau_m)(v), label="$\\tau_m$", c="green")
plt.plot(v, np.vectorize(tau_h)(v), label="$\\tau_h$", c="red")
plt.title("Tiempos Caracteristicos")
plt.legend()
plt.show()


"""
# EJERCICiO 3.1

parametros para las probabilidades de activacion o inactivacion de las
compuertas
"""
v = 0.0  # potencial de la membrana
nt = 0.0  # probabilidad de activacion de K (potasio)
mt = 0.0  # probabilidad de activacion de Na (sodio)
ht = 0.0  # probabilidad de inactivacion Na (sodio)
x = [v, nt, mt, ht]

# parametros para formulas
c = 1  # 1𝜇𝐹/𝑐𝑚2
gna = 120.0  # 120𝑚𝑆/𝑐𝑚2
gk = 36.0  # 36𝑚𝑆/𝑐𝑚2
gl = 0.3  # 0.3𝑚𝑆/𝑐𝑚2
vna = 120.0  # 120𝑚𝑉
vk = -12.0  # −12𝑚𝑉
vl = 10.6  # 10.6𝑚𝑉


def i(t):
    """
    ∼ 10𝜇𝐴/𝑐𝑚2  : corriente de entrada al tiempo 𝑡
    """
    return 0  # por ahora


p = [c, gna, gk, gl, vna, vk, vl, i]


a = 0  # tiempo inicial
b = 500  # tiempo final
h = 0.01  # paso
k = int((b - a) / h)  # canitdad de pasos en el tiempo dado


# 3.1)
t, w = integrador_ode(rk4, f, x, a, b, k, p)
print(w[0, :])


"""
3.2)  Grafique el potencial de membrana en función del tiempo,
    i.e. grafique  𝑣(𝑡)  vs  𝑡  en el rango calculado.
"""

plt.title("Potencial de Membrana")

plt.xlabel("$t$ [ms]")
plt.ylabel("$ v(t) [mV] ")
plt.xlim(0, 50)
plt.ylim(-20, 120)


plt.plot(t, 0 * t, linestyle="--", c="red")
plt.plot(t, w[0], linestyle="-", c="b")
plt.legend()
plt.show()

"""

3.3) Grafique las diferentes corrientes de iones cruzando la membrana en
    función del tiempo, i.e. grafique la corriente de iones de sodio
    𝑖𝑁𝑎(𝑡) , la corriente de iones de potasio  𝑖𝐾(𝑡)  y la corriente de pérdida
    𝑖𝑙(𝑡)  vs  𝑡  en el rango calculado.
"""

plt.title("Corrientes")

plt.xlabel("$t$ [ms]")
plt.ylabel("Micro Amperes ")
plt.xlim(0, 50)
plt.ylim(-30, 30)

plt.plot(t, 0 * t, linestyle="--", c="gray")
plt.plot(t, [ina(w[:, j], p) for j in range(len(t))], label="i_Na", c="r")
plt.plot(t, [ik(w[:, j], p) for j in range(len(t))], label="i_K", c="b")
plt.plot(t, [il(w[:, j], p) for j in range(len(t))], label="i_l", c="g")

plt.legend()
plt.show()

"""
3.4) Grafique las fracción de canales activados e inactivados de cada
    tipo en función del tiempo, i.e. grafique  𝑛(𝑡)
    (fracción de canales de potasio  𝐾  activos),  𝑚(𝑡)
    (fracción de canales de sodio  𝑁𝑎  activos) y  ℎ(𝑡)
    (fracción de canales de sodio  𝑁𝑎  inactivos)
    vs  𝑡  en el rango calculado.
"""
plt.title("Fraccion de canales activos e inactivos")

plt.xlabel("$t$ [ms]")
plt.xlim(0, 50)
plt.ylim(-0.1, 1.1)


plt.plot(t, 0 * t, linestyle="--", c="gray")
plt.plot(t, 0 * t + 1, linestyle="--", c="gray")

plt.plot(t, w[1], label="$n(t)$", c="r")
plt.plot(t, w[2], label="$m(t)$", c="b")
plt.plot(t, w[3], label="$h(t)$", c="g")

plt.legend()
plt.show()


"""
3.5) Almacene los valores de equilibrio a corriente nula de las variables
    dinámicas usando los valores de las mismas obtenidos a tiempos largos.
    Es decir,
    almacene los valores  𝑣∗≈𝑣(𝑡𝑓) ,  𝑛∗≈𝑛(𝑡𝑓) ,  𝑚∗≈𝑚(𝑡𝑓)  y  ℎ∗≈ℎ(𝑡𝑓) ,
    para ser utilizado como condiciones iniciales en futuras integraciones de
    las ODEs del modelo de Hodgkin y Huxley.
"""
valores_equilibrio = w[:, -1]


"""

Ejercicio 4) estímulo débil y estímulo fuerte

    1) Implemente una función de corriente de entrada o membrana dada por:

    𝑖(𝑡) =
    10𝜇𝐴/𝑐𝑚2, 𝑡∈[2𝑚𝑠,2.5𝑚𝑠]
    30𝜇𝐴/𝑐𝑚2, 𝑡∈[10𝑚𝑠,10.5𝑚𝑠]
    0𝜇𝐴/𝑐𝑚2,𝑐.𝑐.

Esta corriente representa un estímulo débil seguido de uno fuerte.
Grafíque la corriente  𝑖(𝑡)  vs  𝑡  en el rango  𝑡∈[0𝑚𝑠,20𝑚𝑠] .
"""



def i2(t):
    if 2.0 <= t <=2.5:
        return 10.0
    if 10.0 <= t <=10.5:
        return 30.0
    return 0.0



plt.title("Corriente")

plt.xlabel("$t$ [ms]")
plt.ylabel("Micro Amperes ")
plt.xlim(0, 20)
t_i = range(0, 20)
plt.plot(t_i, [i2(_) for _ in t_i], c="cyan")

plt.legend()
plt.show()


"""

4.2) Integre nuevamente el sistema de ODEs del modelo de Hodgkin y Huxley
    sujeto a la corriente del inciso anterior. Use como condición inicial
    a tiempo  𝑡=0  los valores de equilibrio estimados en el
    ejercicio anterior. Integre hasta el tiempo final  𝑡𝑓=500𝑚𝑠
    usando un paso temporal  𝑑𝑡=0.01𝑚𝑠 .
"""




a = 0  # tiempo inicial
b = 500  # tiempo final
h = 0.01  # paso
k = int((b - a) / h)  # canitdad de pasos en el tiempo dado
p = [c, gna, gk, gl, vna, vk, vl, i2]

t, w = integrador_ode(rk4, f, valores_equilibrio, a, b, k, p)
print(w[0, :])


"""
4.3) Grafique el potencial de membrana en función del tiempo, i.e.  𝑣(𝑡)
    vs  𝑡  en el rango calculado.
"""
plt.title("Potencial de Membrana para funcion de corriente i_2")

plt.xlabel("$t$ [ms]")
plt.ylabel("$ v(t) [mV] ")
plt.xlim(0, 20)
plt.ylim(-20, 120)


plt.plot(t, t * 0, linestyle="--", c="red")
plt.plot(t, w[0], linestyle="-", c="b")
plt.legend()
plt.show()

"""
4.4) Grafique la evolución de las fraciones de canales activos e inactivos,
    𝑛(𝑡) ,  𝑚(𝑡)  y  ℎ(𝑡)  vs  𝑡 .
"""

plt.title("Fraccion de canales activos e inactivos para i_2")

plt.xlabel("$t$ [ms]")
plt.xlim(0, 20)
plt.ylim(-0.1, 1.1)


plt.plot(t, 0 * t, linestyle="--", c="gray")
plt.plot(t, 0 * t + 1, linestyle="--", c="gray")

plt.plot(t, w[1], label="$n(t)$", c="r")
plt.plot(t, w[2], label="$m(t)$", c="b")
plt.plot(t, w[3], label="$h(t)$", c="g")

plt.legend()
plt.show()


"""
4.5) Discuta como responde la neurona en el primer impulso a 𝑡=2𝑚𝑠 .
    Luego, como responde al segundo impulso a  𝑡=10𝑚𝑠.
    Existe una diferencia? Explique.
"""


"""
Ejercicio 5) ráfaga
5.1) Implemente la corriente de membrana

𝑖(𝑡)={10𝜇𝐴/𝑐𝑚2,0𝜇𝐴/𝑐𝑚2,𝑡∈[5𝑚𝑠,∞𝑚𝑠)𝑐.𝑐.

Esta corriente representa un estímulo constante.

"""

"""

5.2) Integre nuevamente las ODEs para 𝑡∈[0𝑚𝑠,100𝑚𝑠], usando como condición
    inicial los valores de equilibrio derivados en el inciso 6)
    y un paso de integración 𝑑𝑡=0.01.
"""

"""
5.3) Grafique nuevamente el potencial de membrana en el rango de tiempos
     calculado.
"""

"""
5.4) Grafique nuevamente fracciones de canales activos e inactivos vs el
    tiempo.
"""

"""
5.5) Discuta lo que observa y explique.
"""


"""
Ejercicio 6) período refractario
6.1) Implemente la corriente de membrana

𝑖(𝑡)={10𝜇𝐴/𝑐𝑚2,0𝜇𝐴/𝑐𝑚2,𝑡∈[10𝑚𝑠𝑘,10𝑚𝑠𝑘+2𝑚𝑠],𝑘∈{1,2,3,4,5,...}𝑐.𝑐.
"""

"""
6.2) Integre nuevamente las ODEs para 𝑡∈[0𝑚𝑠,100𝑚𝑠], usando la corriente del
     inciso 12), la condición incial los valores de equilibrio derivados
     en el inciso 6) y un paso de integración 𝑑𝑡=0.01.
"""

"""
6.3) Grafique nuevamente el potencial de membrana en el rango de tiempos
    calculado.
"""


"""
6.4) Grafique nuevamente fracciones de canales activos e inactivos
     vs el tiempo.
"""

"""
6.5) Discuta lo que observa y explique.
"""

"""
Ejercicio 7) exitaciones espontáneas en respuesta al ruido

7.1) Implemente una corriente estocástica que retorne un valor
      𝑖(𝑡)∼𝑖0𝑁(0,1)

     (i.e. 𝑖0 por un valor aleatorio obtenido de una
     distribución normal de media 0 y varianza 1)
     para cada valor de 𝑡 en el que sea evaluada.
"""

"""
7.2) Integre nuevamente las ODEs para 𝑡∈[0𝑚𝑠,500𝑚𝑠], usando la corriente
     del inciso 22) para 𝑖0=50𝜇𝐴, la condición incial los valores de
     equilibrio derivados en el inciso 6) y un paso de integración 𝑑𝑡=0.01.
"""

"""
7.3) Grafique nuevamente el potencial de membrana en el rango de tiempos
     calculado.
"""

"""
7.4) Grafique nuevamente fracciones de canales activos e inactivos vs el
     tiempo.
"""

"""
7.5) Observa picos de activación cada tanto? Aparecen con regularidad?
      Estime con que frecuencia observa los picos.

"""
