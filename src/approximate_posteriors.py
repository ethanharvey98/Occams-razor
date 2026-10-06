import math
import torch

class BaseCovariance(torch.nn.Module):

    def __init__(self, m, sigma_f, sigma_n2, tau, temp, cold):
        super().__init__()
        self.register_buffer("m", m)
        self.register_buffer("sigma_f", sigma_f)
        self.register_buffer("sigma_n2", sigma_n2)
        self.register_buffer("tau", tau)
        self.register_buffer("temp", temp)
        self.register_buffer("cold", cold)
        
    def lml(self, Phi, y):
        N, R = Phi.shape
        norm_const = N * math.log(2 * math.pi)
        Sigma = self.sigma_n2 * torch.eye(N, device=Phi.device) + (self.sigma_f ** 2 * self.tau) * (Phi @ Phi.T)
        Chol_Sigma = torch.linalg.cholesky(Sigma)
        log_det = 2 * torch.sum(torch.log(torch.diag(Chol_Sigma)))
        alpha = torch.cholesky_solve(y, Chol_Sigma)
        quad_term = y.T @ alpha
        lml = -0.5 * (norm_const + log_det + quad_term)
        return lml
    
    def elbo(self, Phi, y):
        N, R = Phi.shape
        norm_const = N * torch.log(2 * torch.pi * self.sigma_n2)
        residual = y - self.sigma_f * Phi @ self.m
        quad_term = (1 / self.sigma_n2) * (torch.sum(residual ** 2) + self.sigma_f ** 2 * self.trace_PhiT_Phi_S(Phi))
        ell = -0.5 * (norm_const + quad_term)
        norm_const = R * math.log(2 * math.pi)
        log_det = R * torch.log(self.tau) 
        trace = (1 / self.tau) * self.trace_S
        quad_term = (1 / self.tau) * torch.sum(self.m ** 2)
        elp = -0.5 * (norm_const + log_det + trace + quad_term)
        norm_const = R * math.log(2 * math.pi)
        log_det = self.log_det_S
        entropy = 0.5 * (norm_const + R + log_det)
        return (1 / (self.temp * self.cold)) * ell + (1 / self.cold) * elp + entropy
    
    def trace_PhiT_Phi_S(self, Phi):
        raise NotImplementedError
        
    @property
    def trace_S(self):
        raise NotImplementedError
        
    @property
    def log_det_S(self):
        raise NotImplementedError
    
    def m_update(self, Phi, y):
        N, R = Phi.shape
        Lambda = (self.sigma_f ** 2 / (self.temp * self.sigma_n2)) * (Phi.T @ Phi) + (1 / self.tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
        Chol_Lambda = torch.linalg.cholesky(Lambda)
        b = (self.sigma_f / (self.temp * self.sigma_n2)) * (Phi.T @ y)
        self.m = torch.cholesky_solve(b, Chol_Lambda)
    
    def sigma_f_update(self, Phi, y):
        Phi_m = Phi @ self.m
        self.sigma_f = torch.sum(y * Phi_m) / (torch.sum(Phi_m ** 2) + self.trace_PhiT_Phi_S(Phi))

    def sigma_n2_update(self, Phi, y):
        N, R = Phi.shape
        residual = y - self.sigma_f * Phi @ self.m
        self.sigma_n2 = (1 / N) * (torch.sum(residual ** 2) + self.sigma_f ** 2 * self.trace_PhiT_Phi_S(Phi))
        
    def tau_update(self, Phi):
        N, R = Phi.shape
        self.tau = (self.trace_S + torch.sum(self.m ** 2)) / R    
        
    def mean(self):
        return self.m
    
    def covariance(self):
        raise NotImplementedError
 
    def sample(self, num_samples):
        loc = self.mean().view(-1)
        covariance_matrix = self.covariance()
        scale_tril = torch.linalg.cholesky(covariance_matrix)
        dist = torch.distributions.MultivariateNormal(loc, scale_tril=scale_tril)
        return dist.sample((num_samples,))

class FullRankCovariance(BaseCovariance):
    
    def __init__(self, m, S, sigma_f, sigma_n2, tau, temp, cold):
        super().__init__(m, sigma_f, sigma_n2, tau, temp, cold)
        self.register_buffer("S", S)
        
    def trace_PhiT_Phi_S(self, Phi):
        return torch.sum((Phi @ self.S) * Phi)
        
    @property
    def trace_S(self):
        return torch.trace(self.S)
        
    @property
    def log_det_S(self):
        Chol_S = torch.linalg.cholesky(self.S)
        return 2 * torch.sum(torch.log(torch.diag(Chol_S)))
              
    def S_update(self, Phi):
        N, R = Phi.shape
        S_inv = (self.sigma_f ** 2 / (self.temp * self.sigma_n2)) * (Phi.T @ Phi) + (1 / self.tau) * torch.eye(R, device=Phi.device, dtype=Phi.dtype)
        Chol_S_inv = torch.linalg.cholesky(S_inv)
        self.S = self.cold * torch.cholesky_inverse(Chol_S_inv)
        
    def step(self, Phi, y):
        self.m_update(Phi, y)
        self.S_update(Phi)
        self.sigma_f_update(Phi, y)
        self.sigma_n2_update(Phi, y)
        
    def covariance(self):
        return self.S
    
class DiagonalCovariance(BaseCovariance):
    
    def __init__(self, m, s, sigma_f, sigma_n2, tau, temp, cold):
        super().__init__(m, sigma_f, sigma_n2, tau, temp, cold)
        self.register_buffer("s", s)
        
    def trace_PhiT_Phi_S(self, Phi):
        return torch.sum((Phi ** 2) * self.s)
        
    @property
    def trace_S(self):
        return torch.sum(self.s)
        
    @property
    def log_det_S(self):
        return torch.sum(torch.log(self.s))
            
    def s_update(self, Phi):
        s_inv = ((1.0 / self.tau) + ((self.sigma_f ** 2 / (self.temp * self.sigma_n2)) * torch.sum(Phi ** 2, dim=0)))
        self.s = self.cold * (1.0 / s_inv)
        
    def step(self, Phi, y):
        self.m_update(Phi, y)
        self.s_update(Phi)
        self.sigma_f_update(Phi, y)
        self.sigma_n2_update(Phi, y)
        
    def covariance(self):
        return torch.diag(self.s)
                
class LowRankPlusDiagonalCovariance(BaseCovariance):
    
    def __init__(self, C, d_2, k, L, m, sigma_f, sigma_n2, tau, temp, cold):
        super().__init__(m, sigma_f, sigma_n2, tau, temp, cold)
        self.register_buffer("C", C)
        self.register_buffer("d_2", d_2)
        self.register_buffer("k", k)
        self.register_buffer("L", L)
          
    def trace_PhiT_Phi_S(self, Phi):
        Phi_1, Phi_2 = Phi[:, :self.k], Phi[:, self.k:]
        Phi_1_L = Phi_1 @ self.L
        Phi_2_C_L = Phi_2 @ self.C @ self.L        
        return torch.sum(Phi_1_L ** 2) + torch.sum(Phi_1_L * Phi_2_C_L) + torch.sum(Phi_1_L * Phi_2_C_L) + torch.sum(Phi_2_C_L ** 2) + torch.sum((Phi_2 ** 2) * self.d_2)
        
    @property
    def trace_S(self):
        return torch.sum(self.L ** 2) + torch.sum((self.C @ self.L) ** 2) + torch.sum(self.d_2)
        
    @property
    def log_det_S(self):
        return 2 * torch.sum(torch.log(torch.diag(self.L))) + torch.sum(torch.log(self.d_2))
            
    def C_update(self, Phi, y):
        N, R = Phi.shape
        Phi_1, Phi_2 = Phi[:, :self.k], Phi[:, self.k:]
        Phi_2T_Phi_2 = Phi_2.T @ Phi_2
        I_R_minus_k = torch.eye(R - self.k, device=Phi.device, dtype=Phi.dtype)
        lhs = (self.sigma_f ** 2 / (self.temp * self.sigma_n2)) * Phi_2T_Phi_2 + (1.0 / self.tau) * I_R_minus_k
        rhs = (self.sigma_f ** 2 / (self.temp * self.sigma_n2)) * (Phi_2.T @ Phi_1)
        Chol_lhs = torch.linalg.cholesky(lhs)
        self.C = -torch.cholesky_solve(rhs, Chol_lhs)
        
    def L_update(self, Phi, y):
        Phi_1, Phi_2 = Phi[:, :self.k], Phi[:, self.k:]
        Phi_C = Phi_1 + Phi_2 @ self.C
        I_k = torch.eye(self.k, device=Phi.device, dtype=Phi.dtype)    
        A = (self.sigma_f ** 2 / (self.temp * self.sigma_n2)) * (Phi_C.T @ Phi_C) + (1.0 / self.tau) * (I_k + self.C.T @ self.C)
        Chol_A = torch.linalg.cholesky(A)
        A_inv = torch.cholesky_inverse(Chol_A)
        self.L = torch.sqrt(self.cold) * torch.linalg.cholesky(A_inv)
        
    def d_2_update(self, Phi, y):
        Phi_1, Phi_2 = Phi[:, :self.k], Phi[:, self.k:]
        d_2_inv = ((1.0 / self.tau) + ((self.sigma_f ** 2 / (self.temp * self.sigma_n2)) * torch.sum(Phi_2 ** 2, dim=0)))
        self.d_2 = self.cold * (1.0 / d_2_inv)
          
    def step(self, Phi, y):
        self.m_update(Phi, y)
        self.C_update(Phi, y)
        self.L_update(Phi, y)
        self.d_2_update(Phi, y)
        self.sigma_f_update(Phi, y)
        self.sigma_n2_update(Phi, y)
            
    def covariance(self):
        V = torch.cat([self.L, self.C @ self.L], dim=0)
        S = V @ V.T
        S[self.k:, self.k:].diagonal().add_(self.d_2)
        return S

class LowRankCovariance(LowRankPlusDiagonalCovariance):
    def __init__(self, C, d_2, k, L, m, sigma_f, sigma_n2, tau, temp, cold, update_tau=False):
        super().__init__(C, d_2, k, L, m, sigma_f, sigma_n2, tau, temp, cold)
        self.update_tau = update_tau

    def step(self, Phi, y):
        self.m_update(Phi, y)
        self.C_update(Phi, y)
        self.L_update(Phi, y)
        self.sigma_f_update(Phi, y)
        self.sigma_n2_update(Phi, y)
        if self.update_tau:
            self.tau_update(Phi)
