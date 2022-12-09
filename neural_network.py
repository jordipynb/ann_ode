import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Perceptron multicapa correspondiente a cada funcion
Nx = nn.Sequential(nn.Linear(2,50), nn.Sigmoid(), nn.Linear(50,1, bias = True))
Ny = nn.Sequential(nn.Linear(2,50), nn.Sigmoid(), nn.Linear(50,1, bias = True))
Nz = nn.Sequential(nn.Linear(2,50), nn.Sigmoid(), nn.Linear(50,1, bias = True))

# Intervalos de la variable independiente
t_min = -2
t_max = 2

# Intervalos de las variables parametricas
a_min = 0
a_max = 2

numbers_of_ti = 100
numbers_of_ai = 100

# Condiciones Iniciales
t0 = 0
x0 = 0
y0 = 0
z0 = 0

# Sistema de Funciones
f = lambda t,x,y,z: -2*t[:,1].unsqueeze(1)*t[:,0].unsqueeze(1)*y
g = lambda t,x,y,z: -2*t[:,1].unsqueeze(1)*t[:,0].unsqueeze(1)*y
h = lambda t,x,y,z: -2*t[:,1].unsqueeze(1)*t[:,0].unsqueeze(1)*y


# Solucion parametrica que garantiza la condicion inicial
Psi_xt = lambda t: x0 + (t[:,0].unsqueeze(1) - t0) * Nx(t)
Psi_yt = lambda t: y0 + (t[:,0].unsqueeze(1) - t0) * Ny(t)
Psi_zt = lambda t: z0 + (t[:,0].unsqueeze(1) - t0) * Nz(t)

criterion = torch.nn.MSELoss()
optimizer_x = torch.optim.LBFGS(Nx.parameters(), lr=0.1)
optimizer_y = torch.optim.LBFGS(Nx.parameters(), lr=0.1)
optimizer_z = torch.optim.LBFGS(Nx.parameters(), lr=0.1)

ti = np.random.uniform(t_min, t_max, numbers_of_ti)
ai = np.random.uniform(a_min, a_max, numbers_of_ai)
t = torch.empty((numbers_of_ti, 0))
t = np.insert(t, t.shape[1], ti, 1)
t = np.insert(t, t.shape[1], ai, 1)
t = torch.Tensor(t)

def loss(t):

    t.requires_grad = True
    outputs_x = Psi_xt(t)
    outputs_y = Psi_yt(t)
    outputs_z = Psi_zt(t)
    grads_x, = torch.autograd.grad(outputs_x, t, grad_outputs=torch.ones_like(outputs_x), create_graph=True)
    grads_y, = torch.autograd.grad(outputs_y, t, grad_outputs=torch.ones_like(outputs_y), create_graph=True)
    grads_z, = torch.autograd.grad(outputs_z, t, grad_outputs=torch.ones_like(outputs_z), create_graph=True)
    Psi_t_x = grads_x[:,0].unsqueeze(1)
    Psi_t_y = grads_y[:,0].unsqueeze(1)
    Psi_t_z = grads_z[:,0].unsqueeze(1)

    return criterion(Psi_t_x + Psi_t_y + Psi_t_z, f(t, outputs_x) + g(t, optimizer_y) + h(t, outputs_z))

def closure_x():
    optimizer_x.zero_grad()
    lx = loss(t)
    lx.backward()
    return lx

def closure_y():
    optimizer_y.zero_grad()
    ly = loss(t)
    ly.backward()
    return ly

def closure_z():
    optimizer_z.zero_grad()
    lz = loss(t)
    lz.backward()
    return lz

erx = torch.inf
ery = torch.inf
erz = torch.inf
while erx > 1e-4 or ery > 1e-4 or erz > 1e-4 :
    erx = optimizer_x.step(closure_x) if erx > 1e-4 else erx
    ery = optimizer_y.step(closure_y) if ery > 1e-4 else ery
    erz = optimizer_z.step(closure_z) if erz > 1e-4 else erz

#### HASTA AQUI LO QUE NECESITAMOS PAR EL EJECUTABLE

print(erx)
print(ery)
print(erz)


a = 1
ti = torch.linspace(-2,2, 100)
ai = np.full(100, a)
tt = torch.empty((100, 0))
tt = np.insert(tt, tt.shape[1], ti, 1)
tt = np.insert(tt, tt.shape[1], ai, 1)
tt = torch.Tensor(tt)

with torch.no_grad():
    yy = Psi_xt(torch.Tensor(tt))

yt = []
for v in ti:
    yt.append(a*torch.exp(-1*v*v))

fig, ax = plt.subplots(dpi = 100)
ax.plot(ti, yt, label = 'True')
ax.plot(ti, yy, '--', label= 'Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$Psi(x)$')
plt.legend(loc = 'best')
plt.show()