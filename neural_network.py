import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

n_f = 3
# Perceptron multicapa correspondiente a cada funcion
Ns = nn.Sequential(
    nn.Linear(1, 100),
    nn.Sigmoid(),
    nn.Linear(100, n_f, bias=True),
)  # el numero de entradas depende de los parametros de las funciones
Ns = Ns.double()

##### NO TOCAR --> ESTO ES PARA CORRERLO EN GPU Y FUNCIONE RAPIDO ###
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print("The model will be running on", device, "device\n")
# Ns.to(device)    # Convert model parameters and buffers to CPU or Cuda
# Ni.to(device)    # Convert model parameters and buffers to CPU or Cuda
# Nr.to(device)    # Convert model parameters and buffers to CPU or Cuda

### FASE DE ENTRENAMIENTO ###
# Intervalos de la variable independiente para entrenar
t_min = 0
t_max = 64

# Intervalos de las variables parametricas
a_min = 1.4
a_max = 1.5

b_min = 0.14
b_max = 0.15

numbers_of_ti = 230
numbers_of_ai = 230
numbers_of_bi = 230

# Condiciones Iniciales
t0 = 0
s0 = 1 - 1e-1
i0 = 1e-1
r0 = 0
N = 1

# Sistema de Funciones
alpha = 1.4247  # Para este caso no es parametrica porq no se busca encontra el mejor alpha o beta
beta = 0.14286  # Para este caso no es parametrica porq no se busca encontra el mejor alpha o beta
S = lambda t, s, i: -alpha * s * i
I = lambda t, s, i: alpha * s * i - beta * i
R = lambda t, s, i: beta * i

# Solucion parametrica que garantiza la condicion inicial
Psi_st = lambda t: s0 + (t[:, 0] - t0) * Ns(t)[:, 0]
Psi_it = lambda t: i0 + (t[:, 0] - t0) * Ns(t)[:, 1]
Psi_rt = lambda t: r0 + (t[:, 0] - t0) * Ns(t)[:, 2]

# Criterio de minimos cuadrados en la funcion de perdida
criterion = torch.nn.MSELoss()
# Descenso gradiente, limitando el valor mínimo y máximo del gradiente
optimizer_s = torch.optim.Rprop(Ns.parameters(), lr=0.005)

# Se inicializa un rango de entrenamiento con valores de t cercanos para que el aprendizaje de la red sea mas objetivo
ti = np.random.uniform(t_min, t_max, numbers_of_ti)
# ai = np.random.uniform(a_min, a_max, numbers_of_ai)
# bi = np.random.uniform(b_min, b_max, numbers_of_bi)
t = torch.empty(
    (numbers_of_ti, 0)
)  # creacion de un tensor con todos los parametros de la funcion
t = np.insert(t, t.shape[1], ti, 1)
# t = np.insert(t, t.shape[1], ai, 1)
# t = np.insert(t, t.shape[1], bi, 1)
t = torch.Tensor(t).double()

# Funcion de Perdida
def loss(t):
    t.requires_grad = True
    outputs_s = Psi_st(t)
    outputs_i = Psi_it(t)
    outputs_r = Psi_rt(t)
    (grads_s,) = torch.autograd.grad(
        outputs_s, t, grad_outputs=torch.ones_like(outputs_s), create_graph=True
    )
    (grads_i,) = torch.autograd.grad(
        outputs_i, t, grad_outputs=torch.ones_like(outputs_i), create_graph=True
    )
    (grads_r,) = torch.autograd.grad(
        outputs_r, t, grad_outputs=torch.ones_like(outputs_r), create_graph=True
    )
    Psi_t_s = grads_s[:, 0]
    Psi_t_i = grads_i[:, 0]
    Psi_t_r = grads_r[:, 0]
    criterio = (
        criterion(Psi_t_s, S(t, outputs_s, outputs_i))
        + criterion(Psi_t_i, I(t, outputs_s, outputs_i))
        + criterion(Psi_t_r, R(t, outputs_s, outputs_i))
    )
    return criterio
    # return criterion(
    #     torch.tensor(0, dtype=torch.float64), criterio
    # )  # se espera que la suma de cada diferencia cuadratica sea lo mas cercano a cero posible


# Propagacion hacia atras
def closure_s():
    # Nuevo conjunto por cada iteracion
    ti = np.random.uniform(t_min, t_max, numbers_of_ti)
    t = torch.empty(
        (numbers_of_ti, 0)
    )  # creacion de un tensor con todos los parametros de la funcion
    t = np.insert(t, t.shape[1], ti, 1)
    t = torch.Tensor(t).double()

    optimizer_s.zero_grad()
    ls = loss(t)
    ls.backward()
    return ls


# Iterando para optimizar
er_s = torch.inf
er_i = torch.inf
er_r = torch.inf
epochs = 10000
for e in range(epochs):
    er_s = optimizer_s.step(closure_s)
    print(f"epoch: {e +1}: {er_s}")
    # er_i = optimizer_i.step(closure_i) if min(er_s, er_i, er_r) > 1e-9 else er_i
    # er_r = optimizer_r.step(closure_r) if min(er_s, er_i, er_r) > 1e-9 else er_r

# Chequeo del tensor
print(er_s)

# Fase de Prueba
# a = 1.4247
# b = 0.14286
ti = torch.linspace(0, 70, 100)
# ai = np.full(100, a)
# bi = np.full(100, b)
tt = torch.empty((100, 0))
tt = np.insert(tt, tt.shape[1], ti, 1)
# tt = np.insert(tt, tt.shape[1], ai, 1)
# tt = np.insert(tt, tt.shape[1], bi, 1)
tt = torch.Tensor(tt)

with torch.no_grad():
    ss = Psi_st(torch.Tensor(tt).double())
    ii = Psi_it(torch.Tensor(tt).double())
    rr = Psi_rt(torch.Tensor(tt).double())


fig, ax = plt.subplots(dpi=100)
ax.plot(ti, ss * N, "--", color="green", label="Aproximacion Susceptible")
ax.plot(ti, ii * N, "--", color="red", label="Aproximacion Infectados")
ax.plot(ti, rr * N, "--", color="black", label="Aproximacion Recuperados")


# Comprobacion con la funcion real
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
    return y  # For odeint


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
