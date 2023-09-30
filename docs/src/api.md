# Gaussian Process.jl

## Kernels
```@docs
GaussianProcess.const_cov_fn 
GaussianProcess.lin_cov_fn
GaussianProcess.noise_cov_fn 
GaussianProcess.ratquad_cov_fn
GaussianProcess.sin_cov_fn 
GaussianProcess.exp_cov_fn 
GaussianProcess.sqexp_cov_fn
```

## GP's
```@docs
GaussianProcess.marginal_lkl
GaussianProcess.latent_GP
GaussianProcess.conditional
GaussianProcess.posterior_predict
```
