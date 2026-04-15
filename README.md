# 42186-Model-Based-Project
Model based project


# Running scripts with UV
First, get all packages using `uv sync`.

Use `uv run -m < path >` to run a script. Do not add .py to the end. Here is an example to run the data loading script

`uv run -m src.data_utils.load_dataset`

# Loading the data
in `src.data_utils.load_dataset`, the function `load_PL_dataset` loads the dataset from kaggle. Add `from data_utils import load_dataset` to the top of scripts to use the function.