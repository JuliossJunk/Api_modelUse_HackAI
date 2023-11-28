
# FastAPI ML Classifier

This module provides an easy way to apply a saved Machine Learning (ML) model for multi-class text classification using FastAPI. The basis of the model was taken: distilbert-base-multilingual-cased


## Features

- Support for various ML models
- Simple API for making predictions
- Built-in support for model saving and loading
- Out-of-the-box support for multi-label classification
- Customizable model hyperparameters
- [Модельки](https://drive.google.com/drive/folders/1dXI-ysKn3pgE2Ji0zsMpp9UZO6yHjq5e?usp=sharing)

## Installation

Install my-project dependencies with pip:

```bash
  pip install -r requirements.txt
```
    
## Usage/Examples
For server run on port 8080

```terminal
uvicorn main:app --reload
```

