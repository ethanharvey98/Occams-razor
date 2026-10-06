import argparse
import os
import random
import itertools
import numpy as np
import pandas as pd
import torch
# Importing our custom module(s)
import approximate_posteriors
import layers
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--approximate_posterior", choices=["FullRankCovariance", "DiagonalCovariance", "LowRankPlusDiagonalCovariance", "LowRankCovariance"], default="FullRankCovariance", type=str)
    parser.add_argument("--cold", default=[1.0], nargs="+", type=float)
    parser.add_argument("--D", default=1, type=int)
    parser.add_argument("--experiment_path", default="results.csv", type=str)
    parser.add_argument("--k", default=[5], nargs="+", type=int)
    parser.add_argument("--kernel", choices=["RBF", "Matern"], default="RBF", type=str)
    parser.add_argument("--N", default=[20], nargs="+", type=int)
    parser.add_argument("--noise", default=[0.1], nargs="+", type=float)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--outputscale", default=[1.0], nargs="+", type=float)
    parser.add_argument("--R", default=[1024], nargs="+", type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--temp", default=[1.0], nargs="+", type=float)
    return parser.parse_args()

def main():
    args = parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.experiment_path:
        os.makedirs(os.path.dirname(args.experiment_path), exist_ok=True)
        
    rows = []

    for k, N, noise, outputscale, R, temp, cold in itertools.product(
        args.k,
        args.N,
        args.noise,
        args.outputscale,
        args.R,
        args.temp,
        args.cold,
    ):
    
        for _ in range(args.num_samples):

            X_numpy = np.random.randn(N, args.D)
            if args.kernel == "RBF":
                K = utils.rbf_kernel(X_numpy, X_numpy, lengthscale=1.0, outputscale=outputscale)
            elif args.kernel == "Matern":
                K = utils.matern_kernel(X_numpy, X_numpy, lengthscale=1.0, nu=2.5, outputscale=outputscale)
            
            #y_numpy = np.sin(3 * np.sum(X_numpy, axis=1, keepdims=True)) + noise * np.random.randn(args.N, 1)
            y_numpy = np.random.multivariate_normal(mean=np.zeros(N), cov=K).reshape(N, 1) + noise * np.random.randn(N, 1)
            
            X = torch.tensor(X_numpy, dtype=torch.float64, device=device)
            y = torch.tensor(y_numpy, dtype=torch.float64, device=device)
                
            if args.kernel == "RBF":
                phi = layers.RBFRFFs(
                    D=args.D, 
                    learnable_lengthscale=False, 
                    learnable_outputscale=False, 
                    lengthscale=1.0, 
                    outputscale=1.0, 
                    R=R, 
                ).to(device)
            elif args.kernel == "Matern":
                phi = layers.MaternRFFs(
                    D=args.D, 
                    learnable_lengthscale=False, 
                    learnable_outputscale=False, 
                    lengthscale=1.0, 
                    nu=2.5, 
                    outputscale=1.0, 
                    R=R, 
                ).to(device)

            if args.approximate_posterior == "FullRankCovariance":
                model = approximate_posteriors.FullRankCovariance(
                    m = None,
                    S = None,
                    sigma_f = torch.tensor([1.0], dtype=torch.float64),
                    sigma_n2 = torch.tensor([1.0], dtype=torch.float64),
                    tau = torch.tensor([1.0], dtype=torch.float64),
                    temp = torch.tensor([temp], dtype=torch.float64),
                    cold = torch.tensor([cold], dtype=torch.float64),
                ).to(device)
            elif args.approximate_posterior == "DiagonalCovariance":
                model = approximate_posteriors.DiagonalCovariance(
                    m = None,
                    s = None,
                    sigma_f = torch.tensor([1.0], dtype=torch.float64),
                    sigma_n2 = torch.tensor([1.0], dtype=torch.float64),
                    tau = torch.tensor([1.0], dtype=torch.float64),
                    temp = torch.tensor([temp], dtype=torch.float64),
                    cold = torch.tensor([cold], dtype=torch.float64),
                ).to(device)
            elif args.approximate_posterior == "LowRankPlusDiagonalCovariance":
                model = approximate_posteriors.LowRankPlusDiagonalCovariance(
                    C = None,
                    d_2 = None,
                    k = torch.tensor(k, dtype=torch.int64),
                    L = None,
                    m = None,
                    sigma_f = torch.tensor([1.0], dtype=torch.float64),
                    sigma_n2 = torch.tensor([1.0], dtype=torch.float64),
                    tau = torch.tensor([1.0], dtype=torch.float64),
                    temp = torch.tensor([temp], dtype=torch.float64),
                    cold = torch.tensor([cold], dtype=torch.float64),
                ).to(device)
            elif args.approximate_posterior == "LowRankCovariance":
                model = approximate_posteriors.LowRankCovariance(
                    C = None,
                    d_2 = 1e-6 * torch.ones((R - k,), dtype=torch.float64),
                    k = torch.tensor(k, dtype=torch.int64),
                    L = None,
                    m = None,
                    sigma_f = torch.tensor([1.0], dtype=torch.float64),
                    sigma_n2 = torch.tensor([1.0], dtype=torch.float64),
                    tau = torch.tensor([1.0], dtype=torch.float64),
                    temp = torch.tensor([temp], dtype=torch.float64),
                    cold = torch.tensor([cold], dtype=torch.float64),
                ).to(device)

            elbos, lmls, best_checkpoint = utils.cavi(X, y, phi, model, args.num_epochs)
            #elbos, lmls, best_checkpoint = utils.cavi_and_adam(X, y, phi, model, args.num_epochs)
            
            phi.load_state_dict(best_checkpoint["phi"])
            model.load_state_dict(best_checkpoint["model"])
            Phi = phi(X)
            
            
            snr = ((outputscale ** 2 * torch.sum(Phi ** 2)) / (N * noise ** 2)).item()
            snr_hat = ((model.sigma_f ** 2 * torch.sum(Phi ** 2)) / (N * model.sigma_n2)).item()
            snr_ratio = snr_hat / snr
            
            PhiT_Phi = Phi.T @ Phi
            prior_variance = 1.0
            A = (outputscale ** 2 / noise ** 2) * PhiT_Phi + (1 / prior_variance) * torch.eye(R, device=Phi.device, dtype=torch.float64)
            b = (outputscale / noise ** 2) * (Phi.T @ y)
            m = torch.linalg.solve(A, b)
            m_ratio = (torch.norm(model.m - m, p=2) / torch.norm(m, p=2)).item()
            
            f_hat = model.sigma_f * Phi @ model.m
            f = outputscale * Phi @ m
            f_ratio = (torch.norm(f_hat - f, p=2) / torch.norm(f, p=2)).item()
            
            S = torch.linalg.inv(A)
            trace_ratio = (torch.trace(PhiT_Phi @ model.covariance()) / torch.trace(PhiT_Phi @ S)).item()

            idx = np.argmax(elbos)
            rows.append({
                "cold": cold,
                "elbo": elbos[idx], 
                "f_ratio": f_ratio, 
                "k": k,
                "lml": lmls[idx],
                "m_ratio": m_ratio,
                "N": N, 
                "noise": noise, 
                "outputscale": outputscale, 
                "R": R, 
                "snr": snr, 
                "snr_hat": snr_hat, 
                "snr_ratio": snr_ratio,
                "temp": temp, 
                "trace_ratio": trace_ratio, 
            })

            if args.experiment_path:
                pd.DataFrame(rows).to_csv(args.experiment_path, index=False)
        
if __name__ == "__main__":
    main()