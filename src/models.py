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
    def __init__(self, C, d_2, k, L, m, sigma_f, sigma_n2, tau, temp, cold):
        super().__init__(C, d_2, k, L, m, sigma_f, sigma_n2, tau, temp, cold)

    def step(self, Phi, y):
        self.m_update(Phi, y)
        self.C_update(Phi, y)
        self.L_update(Phi, y)
        self.sigma_f_update(Phi, y)
        self.sigma_n2_update(Phi, y)
        
        
class EmpiricalBayesMixin:
    def step(self, Phi, y):
        super().step(Phi, y)
        self.tau_update(Phi)

        
class FullRankCovarianceEmpiricalBayesMixin(EmpiricalBayesMixin, FullRankCovariance): pass


class DiagonalCovarianceEmpiricalBayesMixin(EmpiricalBayesMixin, DiagonalCovariance): pass


class LowRankPlusDiagonalCovarianceEmpiricalBayesMixin(EmpiricalBayesMixin, LowRankPlusDiagonalCovariance): pass


class LowRankCovarianceEmpiricalBayesMixin(EmpiricalBayesMixin, LowRankCovariance): pass


# TODO: class HGP(torch.nn.Module):


# TODO: class VHGP(torch.nn.Module):


# TODO: class WSHGP(torch.nn.Module):


class WSVHGP(torch.nn.Module):    

    def __init__(self, m_g, mu_0, raw_L_g):
        super().__init__()
        self.m_g = torch.nn.Parameter(m_g)
        self.mu_0 = torch.nn.Parameter(mu_0)
        self.raw_L_g = torch.nn.Parameter(raw_L_g)

    @staticmethod
    def _to_chol(raw_L):
        return torch.tril(raw_L, diagonal=-1) + torch.diag(torch.nn.functional.softplus(torch.diagonal(raw_L)))

    @property
    def L_g(self):
        return self._to_chol(self.raw_L_g)

    def q_fstar(self, Phi_f, Phi_g, y):
        var_g = ((Phi_g @ self.L_g) ** 2).sum(dim=1, keepdim=True)
        r = torch.exp(self.mu_0 + Phi_g @ self.m_g - 0.5 * var_g)
        A = torch.eye(Phi_f.shape[1], dtype=Phi_f.dtype, device=Phi_f.device) + Phi_f.T @ (Phi_f / r)
        C = torch.linalg.cholesky(A)
        m_f = torch.cholesky_solve(Phi_f.T @ (y / r), C)
        S_f = torch.cholesky_inverse(C)
        L_f = torch.linalg.cholesky(S_f)
        return m_f, L_f

    def elbo(self, Phi_f, Phi_g, y):
        m_f, L_f = self.q_fstar(Phi_f, Phi_g, y)
        mean_f = Phi_f @ m_f
        var_f = ((Phi_f @ L_f) ** 2).sum(dim=1, keepdim=True)
        mean_g = Phi_g @ self.m_g
        var_g = ((Phi_g @ self.L_g) ** 2).sum(dim=1, keepdim=True)
        sq_err = (y - mean_f) ** 2 + var_f
        exp_term = torch.exp(-self.mu_0 - mean_g + 0.5 * var_g)
        ell = (-0.5 * math.log(2 * math.pi) - 0.5 * (self.mu_0 + mean_g) - 0.5 * sq_err * exp_term).sum()
        kl_f = 0.5 * ((L_f ** 2).sum() + (m_f ** 2).sum() - m_f.shape[0] - 2.0 * torch.log(torch.diagonal(L_f)).sum())
        kl_g = 0.5 * ((self.L_g ** 2).sum() + (self.m_g ** 2).sum() - self.m_g.shape[0] - 2.0 * torch.log(torch.diagonal(self.L_g)).sum())
        return ell - kl_f - kl_g

    @torch.no_grad()
    def predict(self, Phi_f_star, Phi_g_star, Phi_f, Phi_g, y):
        m_f, L_f = self.q_fstar(Phi_f, Phi_g, y)
        mean = Phi_f_star @ m_f
        epistemic_var = ((Phi_f_star @ L_f) ** 2).sum(dim=1, keepdim=True)
        mean_g = Phi_g_star @ self.m_g
        var_g = ((Phi_g_star @ self.L_g) ** 2).sum(dim=1, keepdim=True)
        aleatoric_var = torch.exp(self.mu_0 + mean_g + 0.5 * var_g)
        return mean, epistemic_var + aleatoric_var
    
    
