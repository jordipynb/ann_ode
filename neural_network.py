import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

n_in = 3   # El número de entradas depende de los parámetros de las funciones
n_out = 3  # El número de salidas corresponde a la cantidad de ecuaciones del sistema

# Perceptrón multicapa
Ns = nn.Sequential(
    nn.Linear(  3, 100, bias=True),
    nn.Sigmoid(),
    nn.Linear(100, n_out, bias=True),
)
Ns = Ns.double()

### FASE DE ENTRENAMIENTO ###
# Intervalos de la variable independiente para entrenar
t_min = 0
t_max = 60

# Intervalos de las variables parametricas
a_min = 1.42
a_max = 1.43

b_min = 0.142
b_max = 0.143

# Cantidad de valores a entrenar dentro del rango correspondiente
numbers_of_ti = 230
numbers_of_ai = 230
numbers_of_bi = 230

# Condiciones Iniciales
t0 = 0
s0 = 1 - 1e-1
i0 = 1e-1
r0 = 0
N = 1

# Sistema de Funciones (Derivadas)
S = lambda t, s, i: -t[:, 1] * s * i
I = lambda t, s, i: t[:, 1] * s * i - t[:, 2] * i
R = lambda t, s, i: t[:, 2] * i

# Solución paramétrica que garantiza la condición inicial
Psi_s = lambda t: s0 + (t[:, 0] - t0) * Ns(t)[:, 0]
Psi_i = lambda t: i0 + (t[:, 0] - t0) * Ns(t)[:, 1]
Psi_r = lambda t: r0 + (t[:, 0] - t0) * Ns(t)[:, 2]

# Criterio de mínimos cuadrados en la función de pérdida
criterion = torch.nn.MSELoss()
# Descenso gradiente (Propagación hacia atrás resiliente)
optimizer_s = torch.optim.Rprop(Ns.parameters(), lr=0.005)

# Se inicializa un rango de entrenamiento con valores de t cercanos para que el aprendizaje de la red sea más objetivo
ti = np.random.uniform(t_min, t_max, numbers_of_ti)
ai = np.random.uniform(a_min, a_max, numbers_of_ai)
bi = np.random.uniform(b_min, b_max, numbers_of_bi)
t = torch.empty(
    (numbers_of_ti, 0)
)  
# Creación de un tensor con todos los parámetros de las funciones
t = np.insert(t, t.shape[1], ti, 1)
t = np.insert(t, t.shape[1], ai, 1)
t = np.insert(t, t.shape[1], bi, 1)
t = torch.Tensor(t).double()

# Función de Pérdida
def loss(t):
    t.requires_grad = True
    outputs_s = Psi_s(t)
    outputs_i = Psi_i(t)
    outputs_r = Psi_r(t)
    (grads_s,) = torch.autograd.grad(
        outputs_s, t, grad_outputs=torch.ones_like(outputs_s), create_graph=True
    )
    (grads_i,) = torch.autograd.grad(
        outputs_i, t, grad_outputs=torch.ones_like(outputs_i), create_graph=True
    )
    (grads_r,) = torch.autograd.grad(
        outputs_r, t, grad_outputs=torch.ones_like(outputs_r), create_graph=True
    )
    Psi_s_t = grads_s[:, 0]
    Psi_i_t = grads_i[:, 0]
    Psi_r_t = grads_r[:, 0]
    criterio = (
          criterion(Psi_s_t, S(t, outputs_s, outputs_i))
        + criterion(Psi_i_t, I(t, outputs_s, outputs_i))
        + criterion(Psi_r_t, R(t, outputs_s, outputs_i))
    ) # Suma de los mínimos cuadrados entre la evaluación en la derivada y los valores de la red por cada ecuación
    return criterio

# Clausura (uso del optimizador)
def closure_s():
    # Nuevo conjunto por cada iteración
    ti = np.random.uniform(t_min, t_max, numbers_of_ti)
    ai = np.random.uniform(a_min, a_max, numbers_of_ai)
    bi = np.random.uniform(b_min, b_max, numbers_of_bi)
    t = torch.empty(
        (numbers_of_ti, 0)
    )  
    # Creación de un tensor con todos los párametros de las funciones
    t = np.insert(t, t.shape[1], ti, 1)
    t = np.insert(t, t.shape[1], ai, 1)
    t = np.insert(t, t.shape[1], bi, 1)
    t = torch.Tensor(t).double()

    optimizer_s.zero_grad()
    ls = loss(t)  # Evaluación en la función de pérdida
    ls.backward() # Propagación hacia atrás resiliente
    return ls

# Iterando para optimizar
er_s = torch.inf
epochs = 10000
for e in range(epochs):
    er_s = optimizer_s.step(closure_s)
    print(f"epoch: {e +1}: {er_s}")


# Fase de Prueba
alpha = 1.4247
beta = 0.14286
ti = torch.linspace(0, 60, 100)
ai = np.full(100, alpha)
bi = np.full(100, beta)
tt = torch.empty((100, 0))
tt = np.insert(tt, tt.shape[1], ti, 1)
tt = np.insert(tt, tt.shape[1], ai, 1)
tt = np.insert(tt, tt.shape[1], bi, 1)
tt = torch.Tensor(tt)

with torch.no_grad():
    ss = Psi_s(torch.Tensor(tt).double())
    ii = Psi_i(torch.Tensor(tt).double())
    rr = Psi_r(torch.Tensor(tt).double())


fig, ax = plt.subplots(dpi=100)
ax.plot(ti, ss * N, "--", color="green", label="Aproximación de Susceptibles")
ax.plot(ti, ii * N, "--", color="red", label="Aproximación de Infectados")
ax.plot(ti, rr * N, "--", color="black", label="Aproximación de Recuperados")


# Comprobación con la función real
import scipy.integrate as spi

# Condiciones Iniciales
alpha = 1.4247
beta = 0.14286
s0 = 1 - 1e-1
i0 = 1e-1
t0 = 0.0
N = 1
input = (s0, i0, t0)

def diff_eqs(INP, t):
    y = np.zeros((3))
    s, i, _ = INP
    y[0] = -alpha * s * i
    y[1] = alpha * s * i - beta * i
    y[2] = beta * i
    return y

t_start = 0.0
t_end = t_max
t_inc = 1.0
t_range = np.arange(t_start, t_end + t_inc, t_inc)
sir = spi.odeint(diff_eqs, input, t_range)

# Gráfica Real
ax.plot(sir[:, 0] * N, "-g", label="Susceptibles")
ax.plot(sir[:, 1] * N, "-r", label="Infectados")
ax.plot(sir[:, 2] * N, "-k", label="Recuperados")
plt.title("Modelo SIR")
ax.set_xlabel("$tiempo$")
ax.set_ylabel("$Personas(\%)$")
plt.legend(loc="best")
plt.show()