
# Precipitation Prediction Transformer (PrecipNet)

[![Build Status](https://img.shields.io/travis/com/your-username/PrecipNet.svg?style=flat-square)](https://travis-ci.com/your-username/PrecipNet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

(PrecipNet](https://www.sciencedirect.com/science/article/pii/S2214581825005671) is a transformer-based deep learning framework for high-resolution precipitation prediction. This repository contains the official implementation for the paper, **"PrecipNet: A transformer-based downscaling framework for improved precipitation prediction in San Diego County."**

The model leverages a transformer architecture to perform statistical downscaling, translating coarse meteorological data into fine-grained, accurate precipitation forecasts.

## Project Structure
- `main.py` — Run this file to train and evaluate the model.
- `model.py` — Contains the transformer model.
- `train.py` — Model training utilities.
- `evaluate.py` — Evaluation and prediction functions.
- `dataset.py` — Prepares the data loaders.
- `config.json` — Configuration file.


## Citation
```bash
@article{adibfar2025precipnet,
  title     = {PrecipNet: A transformer-based downscaling framework for improved precipitation prediction in San Diego County},
  author    = {AmirHossein Adibfar and Hassan Davani},
  journal   = {Journal of Hydrology: Regional Studies},
  volume    = {62},
  pages     = {102738},
  year      = {2025},
  month     = {dec}
}
```
## Instructions
```bash
pip install -r requirements.txt
python main.py

