import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

### Red neuronal con un solo parámetro de entrada, una capa oculta con 50 neuronas
### y función de activación: Sigmiode
N = nn.Sequential(nn.Linear(1,50), nn.Sigmoid(), nn.Linear(50,1))

### Condiciones de la ecuación diferecial ordinaria
f = lambda x, Psi: 3*x**2
A = 0

### Solución a entrenar por la red neuronal
Psi_t = lambda x: A + x * N(x)

### Función del error
def loss(x):
    x.requires_grad = True
    outputs = Psi_t(x)

    # La siguiente línea obtiene el resultado de evaluar la derivada de la función a entrenar 
    # en un vector de valores dado
    Psi_t_x = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    return torch.mean((Psi_t_x - f(x, outputs))**2)

### clase para optmizar el resultado del error
optimizer = torch.optim.LBFGS(N.parameters())

### vector inicial
x = torch.Tensor(np.linspace(-1,1,50)[:, None])
### Método decorador para optimizar el error
def closure():
    optimizer.zero_grad()
    l = loss(x)
    l.backward()
    return l

er = 0
for i in range(20):
    er = optimizer.step(closure)
print(er)

### Comparar resultados
xx = np.linspace(-1,1,100)[:, None]

with torch.no_grad():
    yy = Psi_t(torch.Tensor(xx))

yt = xx**3
fig, ax = plt.subplots(dpi = 100)
ax.plot(xx, yt, label = 'True')
ax.plot(xx, yy, '--', label= 'Neural network approximation')
ax.set_xlabel('$x$')
ax.set_ylabel('$Psi(x)$')
plt.legend(loc = 'best')
plt.show()
