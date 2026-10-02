# DNN vs CNN for OFDM Channel Estimation Under Gaussian and Middleton Class-A Noise

A comparative research repository for deep-learning-based OFDM channel estimation using matched Deep Neural Network (DNN) and Convolutional Neural Network (CNN) architectures under additive white Gaussian noise (AWGN) and Middleton Class-A impulsive noise.

This repository contains the project manuscript source and the complete set of result figures supplied with the project archive.

> **Note:** The uploaded archive did not contain executable Python, MATLAB, model-weight, or dataset files. Therefore, this repository presents the documented methodology and reported experimental results, but it is not currently an executable reproduction package.

## Study Overview

The project compares DNN and CNN estimators for frequency-domain OFDM channel estimation while keeping their trainable parameter counts approximately matched.

The documented simulation setup uses:

- 64 OFDM subcarriers
- Cyclic prefix length: 16
- QPSK modulation
- 16 comb-type pilots, placed every fourth subcarrier
- 8-tap multipath channel with exponential power-delay profile
- SNR range: -5 to 30 dB
- AWGN and Middleton Class-A impulsive noise
- 100,000 training samples
- Adam optimization
- 120 training epochs
- MSE training loss

The manuscript reports approximately 432K trainable parameters for the DNN and 418K for the CNN.

## Architectures

### DNN

The documented DNN processes real and imaginary pilot estimates as a flattened input vector. The reported fully connected structure is:

```text
32 -> 256 -> 512 -> 256 -> 2N
```

ReLU activation and batch normalization are used in the hidden layers.

### CNN

The documented CNN treats pilot estimates as a two-channel one-dimensional input. It uses three convolutional layers with:

```text
64 -> 128 -> 64 filters
kernel size = 3
padding = 1
```

The convolutional stack is followed by global average pooling and fully connected layers.

## Reported Results

The values below are reproduced from the supplied project manuscript and have not been independently re-run in this repository.

| Metric at 20 dB SNR | DNN | CNN |
|---|---:|---:|
| NMSE under AWGN | -19.4 dB | -21.8 dB |
| NMSE under Class-A noise | -19.4 dB | -22.5 dB |
| BER under Class-A noise | 1.26 × 10^-2 | 9.1 × 10^-3 |
| MAC operations per OFDM symbol | 2.4 M | 1.8 M |
| Reported convergence epoch | 71 | 43 |

For Middleton Class-A noise with `A = 0.08` and `Gamma = 0.01`, the manuscript reports a 3.1 dB NMSE difference and a 28% BER reduction for CNN relative to DNN at 20 dB SNR.

The manuscript also reports sensitivity experiments over the impulsive-noise parameters and per-subcarrier error analysis.

## Included Figures

### System and Training

![System block diagram](fig01_system_block_diagram.png)

![Training curve](fig01_training_curve.png)

### Noise Characterization

![Noise PDF comparison](fig02_noise_pdf_comparison.png)

![Noise time-domain comparison](fig03_noise_time_domain.png)

![Noise tail CCDF](fig12_noise_tail_ccdf.png)

### Channel Model

![Power delay profile](fig04_power_delay_profile.png)

![Channel frequency response](fig05_channel_frequency_response.png)

### Signal and Training Behavior

![Constellation diagrams](fig07_constellation_diagrams.png)

![Training curves](fig10_training_curves.png)

### Performance

![MSE versus SNR](fig12_mse_vs_snr.png)

![BER versus SNR](fig13_ber_vs_snr.png)

![Channel estimation comparison](fig14_channel_estimation_comparison.png)

![Error per subcarrier](fig15_error_per_subcarrier.png)

![Error CDF](fig16_error_cdf.png)

![Error histogram](fig17_error_histogram.png)

### Middleton Parameter Sensitivity

![Impact of A](fig18_impact_of_A.png)

![Impact of Gamma](fig19_impact_of_Gamma.png)

### Complexity and Multi-Metric Comparison

![Complexity comparison](fig20_complexity_comparison.png)

![Radar chart](fig21_radar_chart.png)

![NMSE heatmap](fig22_nmse_heatmap.png)

![Error boxplot](fig23_error_boxplot.png)

![Gaussian versus impulsive noise](fig24_gaussian_vs_impulsive.png)

![Summary table](fig25_summary_table.png)

## Repository Contents

This repository is intentionally kept **flat** for simple GitHub upload.

```text
README.md
PROJECT_MANUSCRIPT.md
LICENSE
.gitignore
fig01_system_block_diagram.png
fig01_training_curve.png
fig02_noise_pdf_comparison.png
fig03_noise_time_domain.png
fig04_power_delay_profile.png
fig05_channel_frequency_response.png
fig07_constellation_diagrams.png
fig10_training_curves.png
fig12_mse_vs_snr.png
fig12_noise_tail_ccdf.png
fig13_ber_vs_snr.png
fig14_channel_estimation_comparison.png
fig15_error_per_subcarrier.png
fig16_error_cdf.png
fig17_error_histogram.png
fig18_impact_of_A.png
fig19_impact_of_Gamma.png
fig20_complexity_comparison.png
fig21_radar_chart.png
fig22_nmse_heatmap.png
fig23_error_boxplot.png
fig24_gaussian_vs_impulsive.png
fig25_summary_table.png
```

## Reproducibility Status

The original archive supplied for this GitHub package contained the manuscript and result figures but no executable implementation.

To turn this into a fully reproducible research repository, future additions should include:

- OFDM simulation source code
- DNN implementation
- CNN implementation
- Middleton Class-A noise generator
- Training and evaluation scripts
- Random seeds
- Model checkpoints
- Exact software versions
- Raw result files used to produce the figures

## Manuscript

The complete Markdown manuscript supplied with the project is available as:

`PROJECT_MANUSCRIPT.md`

The references and literature claims in the manuscript should be verified against the original publications before academic submission.

## License

Repository packaging and project materials are provided under the license in `LICENSE`. Third-party material, if any, remains subject to its original rights and terms.
