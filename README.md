# EMRI_SNR_selection
This GPU-enabled repo can be used to:

1. Rapidly generate time-domain EMRI waveforms with TDI and coloured by LISA's power spectral density
2. Calculate the SNRs of EMRIs, and understand the distribution of SNRs with varying/fixed redshift
3. Filter out EMRIs whose parameters fall outside of the range [20,100]

The result of this is a dataset of EMRI parameters with SNRs that could be realistically detected by the LISA detector. Potential applications for this include signal detection, parameter estimation of EMRIs etc.

Packages/citations (to-do):
FEW, LISAonGPU, cupy, numpy, etc.
