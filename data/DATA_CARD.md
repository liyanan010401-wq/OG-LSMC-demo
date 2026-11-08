# Synthetic data card

## Origin and scope

`sample.npz` is generated entirely by [src/data.py](../src/data.py). Each image
combines a smooth Gaussian component, sinusoidal amplitude variation and a
linear phase. There are no patient scans or external imaging datasets in this
file. The legacy array name `patient_id` holds synthetic identifiers, not real
patient identities. All intensities and Fourier values are normalized,
dimensionless demonstration quantities.

## Arrays

| Key | Shape / type | Meaning |
|---|---|---|
| `images` | `(24, 8, 8)`, complex64 | Self-generated complex geometric images. |
| `mask` | `(8, 8)`, bool | Acquired FFT rows 0, 1, 3, 5 and 7, giving 40 acquired complex coefficients. |
| `sens` | `(1, 8, 8)`, complex64 | One synthetic unit-sensitivity coil. |
| `split` | `(24,)`, strings | 18 train, 5 development-reserved, 1 test. |
| `patient_id` | `(24,)`, integers | Synthetic IDs 0 through 23. |
| `seed` | scalar integer | Data-generator seed 4202. |

IDs 0-17 fit the PCA, ridge operator and linear denoisers. IDs 18-22 are held
aside as development samples and are not used for tuning by the script.
ID 23 is the test image. The test reference is used for post-sampling evaluation.

## Measurement generation

The FFT uses `norm='ortho'`. Only acquired coefficients are returned by the
forward operator; missing coefficients are not treated as zero-valued
measurements. During each run, independent real and imaginary Gaussian noise
with variance 0.01 per coordinate is added to the test observation. Thus the
expected squared magnitude of the complex noise is 0.02 per coefficient.
The noise seed is the configured run seed plus one. The sampling seed is the
configured run seed plus two.

The supplied unit coil is a simulation condition, not an independently
estimated physical sensitivity map. No raw clinical k-space, MRI calibration
data or patient-identifying metadata are included.

## Reproduction

Run `python -m src.data` from the repository root to regenerate the file.
Array values are deterministic for the generator and dependency versions;
NPZ container timestamps can change its byte hash. The actual file hash used
by a run is saved in `metrics.json`. Compare arrays when comparing a regenerated
container, and retain the input hash for the recorded execution.
