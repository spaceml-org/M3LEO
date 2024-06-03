# M3LEO

## Installation

Install Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html) and then run the following commands to create the m3leo-env environment:

```bash
conda env create -f environment.yml

conda activate m3leo-env
```

Next, install the package:

```bash
pip install -e .
```

or if you want development dependencies as well:

```bash
pip install -e .[dev]
```

### Optional, but highly recommended

Install [pre-commit](https://pre-commit.com/) by running the following command to automatically run code formatting and linting before each commit:

```bash
pre-commit install
```

If using pre-commit, each time you commit, your code will be formatted, linted, checked for imports, merge conflicts, and more. If any of these checks fail, the commit will be aborted.

## Adding a new package

To add a new package to the environment, open pyproject.toml file and add the package name to "dependencies" list. Then, run the following command to install the new package:

```bash
pip install -e . # or .[dev]
```

## Cache data
Data will be stored in `.cache` inside the folder from where you run the script. If you want to change the cache_dir location, you can set the environment variable `CACHE_DIR` to the desired location. To do so, create a `.env` file and add inside it the following line:

```bash
CACHE_DIR=/path/to/cache/dir
```

## Running train.py
Our training script is fully hydra integrated. To run experiments, set up configuration files following the example provided under <configs/example-config>.

The training script can then be run using

```bash
python train.py --config-path /path/to/config
```