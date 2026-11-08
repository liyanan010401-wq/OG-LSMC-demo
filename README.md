# OG-LSMC synthetic CPU demonstration

This repository provides a small, runnable demonstration of operator-guided
latent sequential Monte Carlo for an explicit complex-valued acquisition model.
It demonstrates a cached operator center, a normalized Gaussian mixture proposal,
importance corrections, resampling, and particle genealogy.

The input consists of self-generated **8 x 8 complex geometric images**. The
encoder/decoder is fitted PCA, the operator is ridge regression, and the noise
predictors are fitted linear models. These are declared substitutes for the
full learned MRI models. This release is a **synthetic mechanism demonstration**;
it does not reproduce the clinical MRI results or the benchmark tables of the
paper. The full MRI training/inference implementation and trained model weights
are not included.

## Quick start

Use Python 3.12 and a CPU environment. From the repository root:

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on macOS/Linux. Then install the pinned dependencies:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5
python run_demo.py --output outputs/my_run
python -m unittest discover -s tests -v
```

Alternatively, `python -m pip install -r requirements.txt` installs the pinned
packages from the configured package index. Use the PyTorch CPU command above
when a CPU-specific wheel is required. No GPU or private checkpoint is needed.

The input data are included. The script also generates `data/sample.npz` if it
is absent. To regenerate the data explicitly, run `python -m src.data`; this
overwrites that one synthetic input file. Configuration and input paths resolve
relative to the script, so `run_demo.py` may also be invoked by absolute path.
An explicitly supplied relative `--output` is relative to the calling directory.
Use a fresh output directory to preserve previous runs.

## Included data and default configuration

- 24 synthetic identifiers: 18 training, 5 development-reserved, 1 test.
- Complex64 images, one simulated unit-sensitivity coil, 40 of 64 Fourier
  coefficients acquired with an orthonormal FFT.
- Four standardized latent coordinates; 96 particles, four reverse transitions,
  and three outer rounds (`N=96`, `T=4`, `J=3`).
- Mixture probability 0.35 and resampling threshold `ESS < 0.5 * N`.
- Independently simulated real and imaginary measurement noise, each with
  variance 0.01; data seed 4202 and run seed 1202.

The benchmark MRI configuration is separate from these small demonstration
settings. See [the data card](data/DATA_CARD.md), [config.json](config.json), and
[the method/interface map](METHOD_MAP.md) for definitions and simplifications.

## Output files

Every successful invocation writes the complete terminal output:

| File | Contents |
|---|---|
| `reconstruction.npz` | Reference image, noisy acquired measurements, all terminal complex particles, their weights, and weighted magnitude/RSS readouts. |
| `metrics.json` | Synthetic magnitude MSE, acquired-data residual, weight sum, ESS and distinct-root-ancestor fractions for each outer round, configuration-related provenance, and the input file hash. |
| `particle_trace.json` | Per-particle log-density terms, chosen mixture component and parent indices for the final outer round at `t < T`. |

The reference image is used for evaluation after sampling; it is not supplied to
the test-time sampler. No selection based on reference error is performed. The
reported `seconds` includes fitting the small surrogate models and generating
the observation, and is not an MRI inference benchmark.

`ess_before_resampling` is the ESS of the cloud represented by that row, before
any subsequent resampling of that cloud. For a transition row, `resampled`
describes the selection of its parents. The initial root-ancestor fraction is
one. The last weighted cloud is read out without terminal resampling. Magnitude
and RSS are computed per particle before averaging with the terminal weights.

The log file records the full marginal proposal density. It does not contain
the initial `t=T` per-particle density decomposition, separate component-density
arrays, or the sampled terminal path index. Those fields must not be inferred
as if they had been exported. The final cloud is conditional on the last
auxiliary draw; finite particles and finite rounds are not a calibration claim.

## Verification and recorded example

The existing ten tests cover the multi-coil complex adjoint, mixture density
and zero-mixture limit, Gaussian normalization and positive variance, gradient
sign, three-boundary telescoping identity, resampling correction, nonfinite
failure handling, frozen-model input derivatives, readout order, and a
fixed-auxiliary Gaussian conditional-moment check.

An actual fresh execution is included under
[`examples/run_20261108`](examples/run_20261108). Its outputs are synthetic
demonstration results. See [`examples/VALIDATION.json`](examples/VALIDATION.json)
for the environment, test count and numerical checks. Floating-point results
and timings may vary across platforms; the recorded weight-sum tolerance is
`1e-5`. These tests check the implemented contracts, not clinical effectiveness
or a nonlinear MRI posterior guarantee.

## Repository contents

```text
run_demo.py              Command-line entry point
config.json              Demonstration settings
requirements.txt         Pinned NumPy and PyTorch versions
src/core.py              Acquisition, surrogate models, densities and sampler
src/data.py              Deterministic synthetic-data generator
tests/test_core.py       Existing mechanism checks
data/sample.npz          Generated synthetic input
data/DATA_CARD.md        Data and split definitions
METHOD_MAP.md            Mechanisms, interfaces and replacement boundaries
examples/                Recorded synthetic execution and verification
NOTICE.md                Release scope and rights notice
```

No patient images, private model weights, private numerical result tables, or
manuscript files are included in this release.
