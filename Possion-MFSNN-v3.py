#!/usr/bin/env python
# coding: utf-8

# <div style="font-size: 80%;">
# 
# 1D Possion equation:
# 
# $\qquad$ $u_{xx} = f(x)$, $\quad$ $x \in [0, 1]$
# 
# $\qquad$ $u(0) = u(1) = 0$,
# 
# Exact solution: $u(x) = a_{1}\sin(\omega_{1}\pi x)+a_{2}\sin(\omega_{2}\pi x)$

# In[ ]:


# Import necessary packages
import time, torch
import torch.nn as nn
import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# Set random seed for reproducibility
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
setup_seed(12)

# CUDA support 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Precision selection
use_double = True  # set 'False' denote 'float32'
dtype = torch.float64 if use_double else torch.float32
print(f'Using dtype: {dtype}')


# In[13]:


# Parameters of equations
a = 1;      b = 0.5;     c = 0.1
w1 = 2;     w2 = 20;     w3 = 200
x_min = 0;  x_max = 1

# Parameters of neural network
N_res = 3000
N_bcs = 1
sigma_1 = 1;  sigma_2 = 7;  sigma_3 = 67

M_res = 3500

inputs_size = 1
hidden_size = 100
concat_size = 300
subdim_size = 1200
output_size = 1

nIter = 60001
eL = 1e-3

# Define the exact solution and its derivatives
def exact(x):
    u = a*torch.sin(w1*torch.pi*x) + b*torch.sin(w2*torch.pi*x) + c*torch.sin(w3*torch.pi*x)
    return u

def terms(x):            
    f = -a*(w1**2)*(torch.pi**2)*torch.sin(w1*torch.pi*x) - b*(w2**2)*(torch.pi**2)*torch.sin(w2*torch.pi*x) - c*(w3**2)*(torch.pi**2)*torch.sin(w3*torch.pi*x)
    return f


# In[14]:


# Calculate the error
def compute_error(model, exact, x):
    u_predi = model(x).detach().cpu().numpy()
    u_exact = exact(x).detach().cpu().numpy()
    l2_error = np.linalg.norm(u_predi-u_exact, 2) / np.linalg.norm(u_exact, 2)
    li_error = np.max(np.abs(u_predi-u_exact))
    return li_error, l2_error

# Create residual sampler
def sample_res(x_min, x_max, N):
    x = torch.linspace(x_min, x_max, N).to(dtype).to(device)
    x = x.reshape(-1, 1)
    return x.requires_grad_(True)

# Create boundary conditions samplers
def sample_bcl(x_min, x_max, N):
    x = torch.ones(N, 1).to(dtype).to(device) * x_min
    return x.requires_grad_(True)

def sample_bcr(x_min, x_max, N):
    x = torch.ones(N, 1).to(dtype).to(device) * x_max
    return x.requires_grad_(True)


# In[15]:


# Define the gradient function
def gradients(u, x, order=1):
    if order==1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order-1)

# Create residual loss function
def compute_loss_res(model, x):
    u_predi = model(x)
    N_res = gradients(u_predi, x, 2)
    E_res = terms(x)
    return torch.mean((N_res - E_res)** 2)

# Create boundary condition loss function
def compute_loss_bcs(model, x):
    N_bcs = model(x)
    E_bcs = exact(x)
    return torch.mean((N_bcs - E_bcs)** 2)


# In[16]:


# Model parameter statistics and memory usage estimation"
def get_n_params(model):
    pp=0
    mem_bytes = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn

        mem_bytes += nn * p.element_size()

    mem_mb = mem_bytes / (1024 ** 2)
    return pp, mem_mb

# Normalization
class NormalizeLayer(torch.nn.Module):
    def __init__(self, a, b):
        super(NormalizeLayer, self).__init__()
        self.register_parameter('a', torch.nn.Parameter(torch.tensor(a, dtype=dtype, requires_grad=False)))
        self.register_parameter('b', torch.nn.Parameter(torch.tensor(b, dtype=dtype, requires_grad=False)))

    def forward(self, x):
        a = self.a.to(device)
        b = self.b.to(device)
        x = x.to(device) 
        return a * x + b

# Random Fourier Feature Map
class RFFeatureMap(torch.nn.Module):
    def __init__(self, N_input, N_hidden, sigma):
        super(RFFeatureMap, self).__init__()
        self.B = torch.nn.Parameter((torch.randn(N_input, N_hidden//2)*sigma).to(dtype), requires_grad=False).to(device)

    def forward(self, x):
        return torch.cat([torch.cos(torch.matmul(x, self.B)), torch.sin(torch.matmul(x, self.B))], dim=-1)

# Creating Neural Network
class MFSNN(nn.Module):
    def __init__(self, inputs_size, hidden_size, concat_size, subdim_size, output_size, a=2/(x_max - x_min), b=-1 - 2*x_min/(x_max - x_min)):
        super(MFSNN, self).__init__()
        self.normalize = NormalizeLayer(a, b)
        self.RFFeatureMap1 = RFFeatureMap(inputs_size, hidden_size, sigma_1)
        self.RFFeatureMap2 = RFFeatureMap(inputs_size, hidden_size, sigma_2)
        self.RFFeatureMap3 = RFFeatureMap(inputs_size, hidden_size, sigma_3)

        self.l0 = nn.Linear(hidden_size, hidden_size, bias=True).to(dtype)
        self.l2 = nn.Linear(hidden_size, hidden_size, bias=True).to(dtype)
        self.l4 = nn.Linear(hidden_size, hidden_size, bias=True).to(dtype)
        self.l6 = nn.Linear(hidden_size, hidden_size, bias=True).to(dtype)
        self.l8 = nn.Linear(concat_size, subdim_size, bias=True).to(dtype)
        self.l10 = nn.Linear(subdim_size, output_size, bias=False).to(dtype)
        self.tanh = nn.Tanh()

    def forward(self, x, subspace=False):
        x = self.normalize(x)
        x1 = self.RFFeatureMap1(x)
        x1 = self.l0(x1);  x1 = self.tanh(x1)
        x1 = self.l2(x1);  x1 = self.tanh(x1)
        x1 = self.l4(x1);  x1 = self.tanh(x1)
        x1 = self.l6(x1);  x1 = self.tanh(x1)
        x2 = self.RFFeatureMap2(x)
        x2 = self.l0(x2);  x2 = self.tanh(x2)
        x2 = self.l2(x2);  x2 = self.tanh(x2)
        x2 = self.l4(x2);  x2 = self.tanh(x2)
        x2 = self.l6(x2);  x2 = self.tanh(x2)
        x3 = self.RFFeatureMap3(x)
        x3 = self.l0(x3);  x3 = self.tanh(x3)
        x3 = self.l2(x3);  x3 = self.tanh(x3)
        x3 = self.l4(x3);  x3 = self.tanh(x3)
        x3 = self.l6(x3);  x3 = self.tanh(x3)

        x = torch.cat([x1, x2, x3], dim=-1)
        x = self.l8(x);  x = self.tanh(x)

        acti_out = x
        if subspace:
            return acti_out

        x = self.l10(x)
        return x

    def subdim_out(self, x):
        return self.forward(x, subspace=True)

model = MFSNN(inputs_size, hidden_size, concat_size, subdim_size, output_size).to(dtype).to(device)

# Model parameter statistics and memory usage estimation
num_params, mem_mb = get_n_params(model)
print(f'Number of parameters: {num_params};  Memory usage: {mem_mb:.2f} MB')


# In[ ]:


# loss logger  
loss_log = []
loss_res_log = []
loss_ics_log = []

x_res = sample_res(x_min, x_max, N_res)  
x_bcl = sample_bcl(x_min, x_max, N_bcs)
x_bcr = sample_bcr(x_min, x_max, N_bcs)
x_bcs = torch.vstack([x_bcl, x_bcr])

x_eva = sample_res(x_min, x_max, M_res)

params_to_optimize = [param for name, param in model.named_parameters() if 'MFSNN[10]' not in name]
optimizer_Adam = torch.optim.Adam(params=params_to_optimize)
model.l10.weight = torch.nn.Parameter(torch.ones_like(model.l10.weight))

start_train = time.time()
for it in range(nIter):
    loss_res = compute_loss_res(model, x_res)
    loss_ics = compute_loss_bcs(model, x_bcs)
    loss = loss_res

    if it == 0:
        flag = loss * eL
        print('Stopping criteria: %.3e' %(flag.detach().cpu().numpy()))
        print('--------Network training started--------')

    # Backward and optimize 
    optimizer_Adam.zero_grad()
    loss.backward()
    optimizer_Adam.step()

    # Store losses
    loss_log.append(loss.detach().cpu().numpy())
    loss_res_log.append(loss_res.detach().cpu().numpy())
    loss_ics_log.append(loss_ics.detach().cpu().numpy())

    # Print
    if it % 100 == 0:
        li_error, l2_error = compute_error(model, exact, x_eva)
        print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_ics: %.3e' %(it, loss.detach().cpu().numpy(), loss_res.detach().cpu().numpy(), loss_ics.detach().cpu().numpy()))
        print('Relative L2 error_u: %.3e, Relative Li error_u: %.3e' %(l2_error, li_error))

    if loss <= flag:
        li_error, l2_error = compute_error(model, exact, x_eva)
        print('--------Network training stopped--------')
        print('It: %d, Loss: %.3e, Loss_res: %.3e,  Loss_ics: %.3e' %(it, loss.detach().cpu().numpy(), loss_res.detach().cpu().numpy(), loss_ics.detach().cpu().numpy()))
        print('Relative L2 error_u: %.3e, Relative Li error_u: %.3e' %(l2_error, li_error))
        break
end_train = time.time()
print('----------------------------------------')
print('Training time: %s(s).' %(end_train - start_train), end='\n\n')


# In[18]:


# Training Loss
loss = loss_log
loss_res = loss_res_log
loss_ics = loss_ics_log

np.save('./data/loss-MFSNN', loss)

fig = plt.figure(figsize=(4, 3.5))
plt.plot(loss, label=r'$\mathcal{L}$')
plt.plot(loss_res, label=r'$\mathcal{L}_{r}$')
plt.plot(loss_ics, label=r'$\mathcal{L}_{u}$')
plt.yscale('log')
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
# plt.savefig(f'Loss.eps', dpi=100)
# plt.show()


# In[19]:


# Plot
loss_res = compute_loss_res(model, x_res)
loss = loss_res
print('Loss: %e' %(loss.detach().cpu().numpy()))

u_predi_numpy = model(x_eva).detach().cpu().numpy()
u_exact_numpy = exact(x_eva).detach().cpu().numpy()
u_error_numpy = np.abs(u_exact_numpy - u_predi_numpy)

l2_error = np.linalg.norm(u_predi_numpy-u_exact_numpy, 2) / np.linalg.norm(u_exact_numpy, 2)
li_error = np.max(np.abs(u_predi_numpy-u_exact_numpy))
print('Relative L2 error_u: %e' % (l2_error)) 
print('Relative Li error_u: %e' % (li_error))

x_eva_numpy = x_eva.detach().cpu().numpy()

fig = plt.figure(figsize=(4, 3.5))
plt.plot(x_eva_numpy, u_predi_numpy)
# plt.xlabel('x', fontsize=10)
plt.title('Prediction', fontsize=10) 
plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(4, 3.5))
plt.plot(x_eva_numpy, u_exact_numpy)
# plt.xlabel('x', fontsize=10)
plt.title('Exact Solution', fontsize=10) 
plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(4, 3.5))
plt.plot(x_eva_numpy, u_error_numpy)
# plt.xlabel('x', fontsize=10)
plt.title('Absolute Error', fontsize=10) 
plt.tight_layout()
# plt.show()


# In[20]:


# Compute gradients
def compute_gradients(u, x):
    grad1 = [];  grad2 = [];  D = u.shape[1]
    for i in range(D):
        g1 = torch.autograd.grad(outputs=u[:, i], inputs=x, grad_outputs=torch.ones_like(u[:, i]), create_graph=True,  retain_graph=True)[0]
        grad1.append(g1.squeeze().detach().cpu().numpy())

        g2 = torch.autograd.grad(outputs=g1,      inputs=x, grad_outputs=torch.ones_like(g1),      create_graph=False, retain_graph=True if i < D-1 else False)[0]
        grad2.append(g2.squeeze().detach().cpu().numpy())

    grad1_np = np.array(grad1).T
    grad2_np = np.array(grad2).T
    return grad1_np, grad2_np

# Assemble least squares matrix
def assemble_matrix(model, x_res, x_bcs):
    u_predi_res = model.subdim_out(x_res)
    _, grad2 = compute_gradients(u_predi_res, x_res)

    A_I = grad2                               
    F_I = terms(x_res).detach().cpu().numpy()

    A_B = model.subdim_out(x_bcs).detach().cpu().numpy()
    F_B = exact(x_bcs).detach().cpu().numpy()

    A = np.vstack((A_I, A_B))
    F = np.vstack((F_I, F_B))
    return (A, F)

# Least squares solver
def solve_lstsq(A, f):
    c = 1.0
    for i in range(len(A)):
        ratio = c / A[i, :].max()
        A[i, :] *= ratio
        f[i] *= ratio      
    w = lstsq(A, f)[0]
    return w


# In[21]:


# Generate least squares matrix
start_lstsq = time.time()
A, f = assemble_matrix(model, x_res, x_bcs)
end_lstsq = time.time()
print('Matrix assembly time: %s(s).' %(end_lstsq-start_lstsq))
print('Matrix shape: N=%s, M=%s.' %(A.shape))

# Solve least squares problem
start_solve = time.time()
w = solve_lstsq(A, f)
end_solve = time.time()
print('Least squares solving time: %s(s).' %(end_solve-start_solve))

w_tensor = torch.from_numpy(w).to(dtype)
w_tensor = w_tensor.view(1, -1)
model.l10.weight = torch.nn.Parameter(w_tensor)
model = model.to(device)


# In[22]:


# Plot
loss_res = compute_loss_res(model, x_res)
loss = loss_res
print('Loss: %e' %(loss.detach().cpu().numpy()))

u_predi_numpy = model(x_eva).detach().cpu().numpy()
u_exact_numpy = exact(x_eva).detach().cpu().numpy()
u_error_numpy = np.abs(u_exact_numpy - u_predi_numpy)

l2_error = np.linalg.norm(u_predi_numpy-u_exact_numpy, 2) / np.linalg.norm(u_exact_numpy, 2)
li_error = np.max(np.abs(u_predi_numpy-u_exact_numpy))
print('Relative L2 error_u: %e' % (l2_error)) 
print('Relative Li error_u: %e' % (li_error))

x_eva_numpy = x_eva.detach().cpu().numpy()

np.save('./data/predi-v3', u_predi_numpy)
np.save('./data/exact-v3', u_exact_numpy)
np.save('./data/error-v3', u_error_numpy)
np.save('./data/gridx-v3', x_eva_numpy)

fig = plt.figure(figsize=(4, 3.5))
plt.plot(x_eva_numpy, u_predi_numpy)
# plt.xlabel('x', fontsize=10)
plt.title('Prediction', fontsize=10) 
plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(4, 3.5))
plt.plot(x_eva_numpy, u_exact_numpy)
# plt.xlabel('x', fontsize=10)
plt.title('Exact Solution', fontsize=10) 
plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(4, 3.5))
plt.plot(x_eva_numpy, u_error_numpy)
# plt.xlabel('x', fontsize=10)
plt.title('Absolute Error', fontsize=10) 
plt.tight_layout()
# plt.show()


# In[ ]:


Lx = 1
D_x = 500

x_line_torch = torch.arange(D_x) * (Lx / D_x)
x_flat = x_line_torch.reshape(-1, 1).to(dtype)
x_flat = x_flat.requires_grad_(True)

u_predi = model(x_flat).detach().cpu().numpy().flatten()
u_exact = exact(x_flat).detach().cpu().numpy().flatten()
li_error_DFT = np.max(np.abs(u_predi - u_exact))
print('DFT Relative Li error_u: %e' % (li_error_DFT)) 

np.save('./data/predi-DFT-v3', u_predi)
np.save('./data/exact-DFT-v3', u_exact)

