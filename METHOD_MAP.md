# Method and interface map

This document distinguishes implemented sampling mechanisms from the compact
surrogates used to make the demonstration runnable on a CPU. It does not claim
equivalence to the full MRI training pipeline.

| Mechanism | Code entry | Implemented behavior and demonstration simplification |
|---|---|---|
| Complex acquisition and adjoint | `Acquisition.forward`, `adjoint`, `measure_real` | Orthonormal FFT, acquired indices only, conjugated coil combination. The sample supplies a simulated unit coil; there is no ESPIRiT or ACS calibration. The adjoint test separately covers multiple complex coils. |
| Real/imaginary representation and latent coordinates | `real`, `complex_image`, `FrozenModels.encode`, `decode` | Training-only PCA with four standardized latent coordinates. This replaces the full learned autoencoder. |
| Cached operator center | `FrozenModels.operator_center`, `Sampler.__init__` | Ridge regression from the adjoint image to standardized latent coordinates, followed by decode/encode. The center is computed once per input and reused. It is not an additional observation likelihood. |
| Reverse prior and terminal variance | `FrozenModels.eps`, `z0`, `Sampler.prior` | Fitted linear noise predictors, configured alpha schedule and eta. At `t=0`, a separately configured positive endpoint variance is used. |
| Likelihood and auxiliary guidance | `Sampler.logL`, `logh`, `auxiliaries` | Normalized Gaussian densities in real coordinates. Each sweep conditions on one fixed set of auxiliary observations. The physical measurement likelihood occurs at the terminal boundary. |
| Base and operator proposal | `Sampler.proposal` | Observation and auxiliary gradients are evaluated at different points. Gradient descent uses fixed componentwise clipping to [-5, 5], an explicit demonstration simplification. The operator center changes with time and ancestor. |
| Complete mixture denominator | `mixture_logpdf` | Both component densities are evaluated for every sampled candidate and combined with `logaddexp`. The selected component is not used alone as the denominator. `rho=0` has an exact base-proposal branch. |
| Initial, intermediate and terminal weights | `Sampler.sweep` | All calculations use log densities. The initial proposal is the standard-normal prior, so its ratio cancels. Intermediate guidance telescopes; the terminal guidance term is omitted. |
| Resampling correction | `Sampler.sweep` | When parents are drawn with probabilities equal to their normalized weights, `log W - log r = 0`. Without resampling, normalized old log weights are carried forward. Nonfinite log weights raise an error rather than silently becoming uniform. |
| Outer iteration and terminal path | `Sampler.initial_path`, `run` | Operator-centered initialization, three auxiliary-refresh rounds and 96 particles. This is not conditional SMC or a particle-MCMC implementation. |
| Readout and diagnostics | `Sampler.run` | Decode the last full weighted cloud; take per-particle magnitude/RSS before averaging. Record ESS and root genealogy. A zero observation-energy denominator produces a null residual. |

## Replacing the surrogate models

An implementation with full learned models must preserve the normalization,
latent representation and phase conventions of `encode`, `decode`, `eps`,
`z0`, and `operator_center`. The demonstration sampler obtains its latent
dimension from `models.basis.shape[0]`; a general implementation should replace
that PCA-specific access with an explicit `latent_dim` interface. Model
parameters should remain frozen while gradients with respect to inputs remain
available to the proposal.

The ridge operator is fitted against standardized latent coordinates, whereas
the full reconstruction operator can require a complex image-domain training
objective. Replacing that objective must not add a second physical likelihood
or bypass the complete mixture correction. For real MRI, supply acquired
complex coefficients, the acquisition mask and fixed coil maps in a consistent
order; calibrate coil maps using only acquired calibration data. Separate
training, development and test examinations before fitting or normalization.

The current initialization always uses the operator center, and the center is
constructed unconditionally. Changing `mixture_rho` alone does not implement a
four-cell initialization/proposal factorial experiment. Such a comparison
requires independent initialization and proposal switches, cost accounting,
matched seeds and frozen prior/resampling settings.

## Diagnostic coverage and logging boundary

The stored final-round trace includes `logL`, `logh`, `logp`, `logh_parent`,
`logq_marginal`, component choices and parent indices for `t=T-1,...,0`.
It does not export the initial per-particle decomposition at `t=T`, separate
component log densities, or the sampled terminal path index. The aggregate
metrics include all outer rounds. Add explicit logging if those additional
fields are needed; do not reconstruct them as measured records from missing
exports.

The included numerical checks exercise these mechanisms in small synthetic
settings. They do not establish MRI reconstruction accuracy, posterior
calibration, full nonlinear path-density validation or the paper's benchmark
comparisons. The confidential MRI code and checkpoints are outside this release.
