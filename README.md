# Streamflow Prediction Based on Neural Delay Differential Equations

The following is the detailed data and code process for the paper

**Streamflow Prediction Based on Neural Delay Differential Equations**

------

## 📂 Project Structure

```
graphql复制编辑STDDE/
├── controldiffeq/             # Core modules for Delay Differential Equation (DDE) modeling
│   ├── ddefunc.py             # Defines the DDE function
│   ├── ddeutil.py             # DDE utility functions
│   ├── interpolate.py         # Time interpolation module for solving DDEs
│   ├── misc.py                # Miscellaneous utilities
├── data/
│   └── changjiang/            # Dataset folder (e.g., Dongting Lake basin streamflow data)
├── lib/                       # Auxiliary code
│   ├── args.py                # Argument parser
│   ├── datapro.py             # Data preprocessing utilities
│   ├── dataset.py             # Custom PyTorch dataset loaders
│   ├── eval.py                # Evaluation metrics
├── logs/                      # Logs directory (e.g., for training)
├── model/                     # DDE-based model definitions
│   ├── ddeatt.py              # DDE model with attention mechanism
│   ├── model.py               # Wrapper model
│   ├── get_delay.py           # Delay estimation utilities
│   ├── run_model.py           # Training/Testing entry point
├── run.sh                     # Shell script to run the pipeline
```

------

## 🚀 Getting Started

### Requirements

- Python 3.8+
- PyTorch ≥ 1.8
- NumPy
- SciPy
- Matplotlib

Install dependencies with:

```
bash

pip install -r requirements.txt
```

------

## 📦 Data

The dataset is located in the `data/changjiang/` directory and contains time-series streamflow data, spatial relationships, and delay information for stations in the **Changjiang (Dongting Lake basin)** . Below is a detailed description of each file:

 `changjiang_com.npy`

- **Description**: This file contains the raw time-series streamflow data for multiple stations in the basin. It serves as the main input for model training and evaluation.
- **Format**: NumPy array with shape `[T, N，F]`, where:
  - `T` is the number of time steps,
  - `N` is the number of stations.
  - `F` is the number of features (such as,water level and streamflow).

------

`distance.csv`

- **Description**: A CSV file recording the geographic or hydrological distance between each pair of stations.
- **Usage**: Used to construct the spatial graph and estimate time delays between stations.
- **Format**: Square matrix where rows and columns represent station IDs, and values denote pairwise distances (in kilometers or hours).

The historical flow data and the distance matrix can be framed by data processing to obtain the appropriate model input `adj.npy` and the delay between sites `delay.npy`



------

## 🧠 Model

The model is based on Neural Delay Differential Equations (NDDE), where streamflow dynamics are modeled with historical dependencies using:

- Learnable DDE functions (`controldiffeq/ddefunc.py`)
- Delay estimation (`model/get_delay.py`)
- Attention mechanisms (`model/ddeatt.py`)

------

## 🏃‍♂️ Training & Evaluation

To train the model:

```
bash

bash run.sh
```

Alternatively, run directly via Python:

```
bash

python model/run_model.py --config your_config.yaml
```