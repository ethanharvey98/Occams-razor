import torch
import torch.nn.functional as F
# Importing custom module(s)
import utils


class BaseRFFs(torch.nn.Module):

    def __init__(
        self,
        D: int,
        lengthscale: float = 1.0,
        R: int = 1_024,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.D = D
        self.R = R
        self.dtype = dtype
        
        self.register_buffer("weight", self._sample_weight())
        self.register_buffer("bias", self._sample_bias())
        self.raw_lengthscale = torch.nn.Parameter(
            utils.inv_softplus(torch.tensor(lengthscale, dtype=dtype))
        )

    def _sample_weight(self) -> torch.Tensor:
        raise NotImplementedError

    def _sample_bias(self) -> torch.Tensor:
        return 2 * torch.pi * torch.rand(self.R, dtype=self.dtype)

    @property
    def lengthscale(self) -> torch.Tensor:
        return F.softplus(self.raw_lengthscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (2 / self.R) ** 0.5 * torch.cos(
            F.linear(x, (1 / self.lengthscale) * self.weight, self.bias)
        )


class RBFRFFs(BaseRFFs):

    def _sample_weight(self) -> torch.Tensor:
        return torch.randn(self.R, self.D, dtype=self.dtype)


class MaternRFFs(BaseRFFs):

    def __init__(
        self, 
        *args, 
        nu: float = 2.5, 
        **kwargs,
    ):
        self.nu = nu
        super().__init__(*args, **kwargs)

    def _sample_weight(self) -> torch.Tensor:
        Z = torch.randn(self.R, self.D, dtype=self.dtype)
        dof = 2.0 * self.nu
        chi2_dist = torch.distributions.Chi2(torch.tensor(dof, dtype=self.dtype))
        C = chi2_dist.sample((self.R, 1))        
        return Z * torch.sqrt(dof / C)


class ScaleFeatures(torch.nn.Module):

    def __init__(
        self, 
        features: torch.nn.Module, 
        outputscale: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.features = features
        self.raw_outputscale = torch.nn.Parameter(
            utils.inv_softplus(torch.tensor(outputscale, dtype=dtype))
        )

    @property
    def outputscale(self) -> torch.Tensor:
        return F.softplus(self.raw_outputscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.outputscale.sqrt() * self.features(x)
    
    