import math
import copy
import numpy as np
from scipy.spatial.distance import cdist
import torch

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))

def rbf_kernel(X1, X2, lengthscale=1.0, outputscale=1.0):
    sq_dist = cdist(X1 / lengthscale, X2 / lengthscale, metric="sqeuclidean")
    return outputscale ** 2 * np.exp(-0.5 * sq_dist)

def matern_kernel(X1, X2, lengthscale=1.0, nu=2.5, outputscale=1.0):
    dists = cdist(X1 / lengthscale, X2 / lengthscale, metric="euclidean")
    assert nu in [0.5, 1.5, 2.5]
    if nu == 0.5:
        return outputscale ** 2 * np.exp(-dists)
    elif nu == 1.5:
        sqrt3_d = np.sqrt(3.0) * dists
        return outputscale ** 2 * (1.0 + sqrt3_d) * np.exp(-sqrt3_d)
    elif nu == 2.5:
        sqrt5_d = np.sqrt(5.0) * dists
        return outputscale ** 2 * (1.0 + sqrt5_d + (5.0 / 3.0) * (dists ** 2)) * np.exp(-sqrt5_d)  
    
def cavi(X, y, phi, model, num_epochs=1_000):
        
    elbos = []
    lmls = []
    prev_elbo = None
    best_elbo = float("-inf")
    best_checkpoint = None
    
    Phi = phi(X)

    for epoch in range(num_epochs):
        
        with torch.no_grad():
            model.step(Phi, y)
            elbo = model.elbo(Phi, y)
            lml = model.lml(Phi, y)
            
        elbos.append(elbo.item())
        lmls.append(lml.item())
    
        if elbo > best_elbo:
            best_elbo = elbo
            best_checkpoint = {
                "phi": copy.deepcopy(phi.state_dict()),
                "model": copy.deepcopy(model.state_dict()),
            }
            
        if prev_elbo is not None and abs(elbo - prev_elbo) < 1e-10:
            break
            
        prev_elbo = elbo

    return elbos, lmls, best_checkpoint
        
def cavi_and_adam(X, y, phi, model, num_epochs=1_000):
        
    elbos = []
    lmls = []
    prev_elbo = None
    best_elbo = float("-inf")
    best_checkpoint = None
    
    optimizer = torch.optim.Adam(phi.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        
        with torch.no_grad():
            Phi = phi(X)
            model.step(Phi, y)
            
        optimizer.zero_grad()
        Phi = phi(X)
        elbo = model.elbo(Phi, y)
        (-elbo).backward()
        optimizer.step()
        
        with torch.no_grad():
            elbo = model.elbo(Phi, y)
            lml = model.lml(Phi, y)
            
        elbos.append(elbo.item())
        lmls.append(lml.item())
    
        if elbo > best_elbo:
            best_elbo = elbo
            best_checkpoint = {
                "phi": copy.deepcopy(phi.state_dict()),
                "model": copy.deepcopy(model.state_dict()),
            }
            
        if prev_elbo is not None and abs(elbo - prev_elbo) < 1e-10:
            break
            
        prev_elbo = elbo

    return elbos, lmls, best_checkpoint
        
