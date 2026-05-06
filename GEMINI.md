# 42186-Model-Based-Project

This project focuses on building and evaluating Bayesian hierarchical models for English Premier League (EPL) player statistics. It leverages probabilistic programming to analyze player ratings and performance across different positions (Goalkeeper, Defender, Midfielder, Attacker), accounting for team-level variations.

## Technology Stack

- **Language:** Python 3.12+
- **Environment Management:** [uv](https://github.com/astral-sh/uv)
- **Probabilistic Programming:** [Pyro](https://pyro.ai/) (built on [PyTorch](https://pytorch.org/))
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (for preprocessing)
- **Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Data Source:** [Kagglehub](https://github.com/kaggle/kagglehub) (EPL 2025/26 Player Stats)

## Project Structure

- `src/data_utils/`: Utilities for data acquisition and analysis.
  - `load_dataset.py`: Downloads the EPL dataset from Kaggle and caches it in `data/`.
  - `correlation.py`: Analyzes Spearman correlations between performance metrics and player ratings.
  - `make_subdataset.py`: Creates position-specific subsets of the data.
- `src/models/`: Implementation of probabilistic models.
  - `attacker_model.py`: Hierarchical model for forwards.
  - `goalkeeper_model.py`: Probabilistic graphical model for goalkeepers.
  - `defender_model.py`: (In development) Models for defenders.
  - `parameter_recovery.py` & `test_recovery.py`: Scripts for validating model architecture through synthetic data parameter recovery.
- `data/`: Local cache for datasets and processed statistics (ignored by git).

## Getting Started

### Installation

Ensure you have `uv` installed. Sync the dependencies and set up the virtual environment:

```powershell
uv sync
```

### Loading Data

To download the latest dataset from Kaggle and prepare the local CSV:

```powershell
uv run -m src.data_utils.load_dataset
```

### Running Models

Execute models as modules. For example, to run the attacker model:

```powershell
uv run -m src.models.attacker_model
```

### Model Validation

To run parameter recovery tests for the goalkeeper model:

```powershell
uv run -m src.models.test_recovery
```

### Performance Evaluation

To evaluate the predictive accuracy of a model (e.g., the attacker model) on a held-out test set (split 80/20):

```powershell
uv run -m src.models.evaluate_performance attacker --steps 2000 --lr 0.01
```

This script:
1.  Outputs Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared ($R^2$) metrics.
2.  Generates a performance plot (Mean Prediction ± Standard Deviation) in the `plots/` directory (e.g., `plots/attacker_eval_plot.png`).

## Development Conventions

- **Module Execution:** Always run scripts as modules using `uv run -m src.<package>.<module>`.
- **Data Handling:** Use `src.data_utils.load_dataset.load_PL_dataset()` to ensure consistent data access.
- **Modeling Style:**
  - Prefer Stochastic Variational Inference (SVI) with `AutoNormal` guides for initial model training.
  - Use `pyro.plate` for hierarchical structures (e.g., grouping by team).
  - Explicitly pass `num_teams` and `num_features` to models to avoid reliance on global state.
- **Testing:** New models should be validated using parameter recovery scripts in `src/models/test_recovery.py` to ensure the architecture can accurately recover known parameters from synthetic data.
