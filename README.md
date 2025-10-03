
# Precipitation Prediction Transformer

[PrecipNet]([https://doi.org/your-paper-doi-here](https://www.sciencedirect.com/science/article/pii/S2214581825005671#da0005)) is a transformer-based deep learning model to predict precipitation using meteorological data.

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

