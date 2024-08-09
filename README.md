# Probabilistic Seismic Hazard Analysis Notebook

This repository contains a Jupyter notebook that performs probabilistic seismic hazard analysis (PSHA) using Python. The notebook walks through the process of creating earthquake objects, generating magnitudes, assigning seismic sources, and calculating intensity measures based on ground motion models.

## Steps for Seismic Hazard Analysis

1. **Create Earthquake Objects**: Generate earthquake scenarios with associated magnitudes and sources.
2. **Magnitude-Frequency-Distribution (MFD)**: Use the Gutenburg-Richter law to determine the frequency of different magnitudes.
3. **Ground-Motion-Equation (GME) Model**: Calculate median ground-motion intensities and the standard deviations of inter-event and intra-event residuals from the ground motion equation model.
5. **Inter-Event and Intra-Event Residuals**: Generate normalized residuals for the seismic events.
6. **Intensity Measure (IM)**: Compute the intensity measure values for the earthquake scenarios.

### Ground Motion Models Supported
- AbrahamsonSilvaKamai2014

### Intensity Types Supported
- PGA (Peak Ground Acceleration) in units of g (gravity)
- SA (Spectral Acceleration) in units of g at different periods
- PGV (Peak Ground Velocity) in units of cm/sec

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/ZeinabFarah/PSHA.git
```
## Acknowledgments
The ground motion modeling is performed using the [pygmm](https://pythonhosted.org/pygmm/) library.\
Magnitude-frequency distribution is currently limited to the Gutenburg-Richter law implementation.

