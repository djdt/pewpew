# pew²

Pew² is a GUI for importing and processing line-by-line, spot-wise and ablation-time-aligned LA-ICP-MS data using the python library [pewlib](https://github.com/djdt/pewlib).

![pew2](https://github.com/djdt/djdt.github.io/blob/main/img/pewpew_splash_1.4.3.png)

## Supported formats

Agilent
 * Drag-and-drop import of batches (.b directories), with one sample per line
 * Import of both binary and csv data
 * Syncing with Iolite laser logs (one sample per image pattern)

Thermo iCap
  * CSV exports
  * LDR (laser-data-reduction) exports

Perkin Elemer
  * ELAN .xl directories

Nu Intruments
  * Vitesse images

CSV Images and Lines
  * Importer for generic one-file-per-csv style data

ImzML
  * Import wizard for data in the imzML format
  * Mass selection and exploration

Iolite Laser Logs
  * Wizard to sync laser and ICP data
  * Both ActiveView2 and Chromium3 style logs

## Installation

Windows executables are available for each [release](https://github.com/djdt/pewpew/releases).

To install via pip first clone the repository then install as a local package.

```bash
git clone https://github.com/djdt/pewpew
cd pewpew
pip install -r requirements.txt
pip install -e .
```

The program can then be run from the command line.

```bash
pewpew
```

## Documentation

Documentation is available at https://pew2.readthedocs.io


## Publications

* [Pew2: Open-Source Imaging Software for Laser Ablation−Inductively Coupled Plasma−Mass Spectrometry](https://doi.org/10.1021/acs.analchem.1c02138)
