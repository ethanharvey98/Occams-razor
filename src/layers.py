import torch
# Importing our custom module(s)
import utils

class BaseRFFs(torch.nn.Module):
    def __init__(
        self, 
        D: int, 
        learnable_lengthscale: bool = False, 
        learnable_outputscale: bool = False, 
        lengthscale: float = 20.0, 
        outputscale: float = 1.0, 
        R: int = 1_024,
    ):
        super().__init__()
        
        self.D = D
        self.R = R
        
        feature_weight = self._sample_weights()
        self.register_buffer("feature_weight", feature_weight)
        self.register_buffer("feature_bias", 2 * torch.pi * torch.rand(self.R, dtype=torch.float64))
                
        if learnable_lengthscale:
            self.raw_lengthscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(lengthscale, dtype=torch.float64)))
        else:
            self.register_buffer("raw_lengthscale", utils.inv_softplus(torch.tensor(lengthscale, dtype=torch.float64)))
        
        if learnable_outputscale:
            self.raw_outputscale = torch.nn.Parameter(utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float64)))
        else:
            self.register_buffer("raw_outputscale", utils.inv_softplus(torch.tensor(outputscale, dtype=torch.float64)))

    def _sample_weights(self) -> torch.Tensor:
        raise NotImplementedError
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outputscale * (2 / self.R) ** 0.5 * torch.cos(torch.nn.functional.linear(x, (1 / self.lengthscale) * self.feature_weight, self.feature_bias))
                                                                
    @property
    def lengthscale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_lengthscale)
    
    @property
    def outputscale(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_outputscale)


class RBFRFFs(BaseRFFs):
    
    def _sample_weights(self) -> torch.Tensor:
        return torch.randn(self.R, self.D, dtype=torch.float64)

class MaternRFFs(BaseRFFs):
    
    def __init__(
        self, 
        *args, 
        nu: float = 2.5,
        **kwargs,
    ):
        self.nu = nu 
        super().__init__(*args, **kwargs)
        
    def _sample_weights(self) -> torch.Tensor:
        Z = torch.randn(self.R, self.D, dtype=torch.float64)
        chi2_dist = torch.distributions.Chi2(torch.tensor(2.0 * self.nu, dtype=torch.float64))
        C = chi2_dist.sample((self.R, 1))
        return Z * torch.sqrt((2.0 * self.nu) / C)
    