# GainMatch

GainMatch is a Qt desktop tool for interactive gain matching of Maestro `.Spe` spectra.
It aligns one or more mobile spectra to a reference spectrum using a linear channel transform:

- `ch_out = m * ch_in + q`

The app provides automatic peak-based fitting, optional spectrum least-squares fitting, normalization controls, subtraction tools, and export of matched/difference spectra.

## Features

- Load one reference spectrum and multiple mobile spectra
- Two independent matching slots (`Match1`, `Match2`) against the same reference
- Automatic peak detection and peak-pair based calibration (`m`, `q`)
- Optional spectrum LSQ fit mode for calibration refinement
- Manual peak picking and manual peak-pair editing
- Normalization modes: `none`, `peak`, `area`, `integral`
- Subtraction modes:
  - `1 - 2`
  - `2 - 1`
  - `Ref - 1`
  - `Ref - 2`
- Region/range tools for summed counts and uncertainty reporting
- Save outputs to a `matched/` folder next to source files

## Requirements

- Python 3.9+
- `numpy`
- `scipy`
- `PySide6`
- `pyqtgraph`

Install dependencies:

```bash
pip install numpy scipy pyside6 pyqtgraph
```

## Usage

```bash
python gainmatch.py REF.Spe MOB1.Spe [MOB2.Spe ...] [--norm peak]
```

### Arguments

- `FILE.Spe ...`
  - First file is the reference spectrum
  - Remaining files are mobile spectra to match
  - If only one file is provided, it is used as both reference and mobile
- `--norm {none,peak,area,integral}`
  - Initial normalization mode (default: `peak`)

## Outputs

GainMatch writes files in a `matched/` subdirectory near the original spectrum file:

- Matched spectrum 1: `<name>_matched.Spe`
- Matched spectrum 2: `<name>_matched.Spe` (for the spectrum used in Match2)
- Subtracted spectrum: `<name>_subtracted_<mode>.Spe`
- Subtracted 3-column text export: `<name>_subtracted3col_<mode>.txt` with columns `x y dy`

(`mode` is one of `12`, `21`, `ref1`, `ref2`)

## Notes

- Spectrum lengths are padded internally to the maximum loaded channel count.
- Difference spectra are live-time scaled during subtraction and written with updated metadata.
- If Qt dependencies are missing, the script exits with an install hint.
