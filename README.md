# GRANDWin 🛰️  
**Gain-based RFI Analysis using Normalized Deviation with Winsorization**

GRANDWin is an outlier detection and flagging algorithm designed for radio interferometric data, based on winsorization statistics. It is developed as part of an academic research project to improve the calibration quality by identifying and flagging radio frequency interference (RFI) contaminated antenna gain solutions.

## 📘 Background

In radio astronomy, gain calibration is critical for extracting accurate radio signals. However, outliers in the gain solutions—caused by RFI, instrumental errors, or bad observations—can degrade image quality and skew results.

**GRANDWin** applies a robust statistical method using **winsorization** to detect outliers in gain solutions across frequency, time, and antennas. It then enables **automated flagging** of contaminated data directly in the UVFITS visibility files.

---

## 🔁 Workflow Overview

The GRANDWin pipeline consists of the following steps:

1. **Input Parsing**  
   - Reads a CSV file containing observation IDs, observation dates, and pointing centers.
   
2. **Gain Data Access**  
   - Loads gain calibration solutions stored in FITS or HDF5 files for each observation.

3. **Outlier Detection**  
   - Applies a winsorized Z-score method to detect statistically abnormal values in the gain data.
   - Outputs a list of outliers with metadata: observation ID, antenna ID, frequency channel, and time step.

4. **Flagging**  
   - Applies flags to the original UVFITS visibility data using the detected outliers. The condition: if we find one contaminated antenna then we will flag all of baselines related to the antenna
   - Saves the modified UVFITS file with flags applied.

[![DOI](https://zenodo.org/badge/680141359.svg)](https://doi.org/10.5281/zenodo.18297420)

---

## 📂 Repository Structure

```text
GRANDWin/
├── data/                           # Input and output data files
│   └── raw/                        # Original CSV, FITS, HDF5, UVFITS files
        ├── calibration_solutions
        └── uvfits_raw
│   └── processed/                  # Output flagged files and reports
        ├── detected_outliers
        └── uvfits_update
│
├── grandwin/                       # Main package code
│   ├── winsorize_detector.py                # Core winsorized outlier detection logic
│   ├── flagger.py                  # UVFITS flagging based on outliers
│   ├── metadata_parser.py          # Extracts obs info from CSV
│   ├── config.py                   # Global settings and thresholds
│   ├── utils.py                    # Logging and helper functions
│   └── io/                         # I/O handling modules
│       ├── base_reader.py
│       ├── fits_reader.py
│       ├── h5_reader.py
│       ├── uvfits_reader.py
│       └── file_reader_factory.py
│
├── notebooks/                      # Jupyter notebooks for testing
├── scripts/                        # CLI scripts to run the pipeline
│   ├── detect_outliers.py
│   └── flag_outliers.py
│
├── tests/                          # Unit tests for each module
├── requirements.txt                # Python dependencies
├── setup.py                        # Install as a package (optional)
├── README.md                       # Project documentation
└── LICENSE                         # License info (e.g., MIT)

## What's New in v2.0.0
* **Major Update:** Implemented bias correction for the Winsorized statistics (Wilcox 2012).
* **Optimized Parameters:** Default outlier detection is now configured for the phase component at a 2s solution interval with a threshold of $|z| > 10$.
* **Performance:** Achieves >99.5% power reduction and a ~35 dB variance drop in MWA 75-100 MHz auto-power spectra.
