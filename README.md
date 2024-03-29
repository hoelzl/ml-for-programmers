# Machine Learning for Programmers

Source code and other material for the course "Machine Learning for Python
Programmers".

## Description

This repository contains both the source code used during the lectures as well
as templates and solutions for exercises.

## Installation

In order to set up the necessary environment:

1. Create an environment `ml-for-programmers` with the help of [conda]:

   ```bash
   conda env create -f environment.yml
   ```

   If you are on Windows, use the file `environment-win.yml` instead to create
   an environment that contains the support for traditional machine learning
   without support for PyTorch/fastai, since the latter libraries are unlikely
   to install without problems.

2. Activate the new environment with:

   ```bash
   conda activate ml-for-programmers
   ```

> **_NOTE:_**  The conda environment will have ml-for-programmers installed in
> editable mode. Some changes, e.g. in `setup.cfg`, might require you to run
> `pip install -e .` again.

Optional and needed only once after `git clone`:

<!-- markdownlint-disable-next-line -->
3. Install several [pre-commit] git hooks with:

   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```

   and checkout the configuration under `.pre-commit-config.yaml`.

   The `-n, --no-verify` flag of `git commit` can be used to deactivate
   pre-commit hooks temporarily.

<!-- markdownlint-disable-next-line -->
4. Install [nbstripout] git hooks to remove the output cells of committed notebooks with:

   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```

   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in
   `environment.yml` and eventually in `setup.cfg` if you want to ship and
   install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact
   reproduction of your environment with:

   ```bash
   conda env export -n ml-for-programmers -f environment.lock.yml
   ```

   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:

   ```bash
   conda env update -f environment.lock.yml --prune
   ```

## Project Organization

```plain-text
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── ml_for_programmers  <- Actual Python package where the main functionality goes.
│       ├── data            <- Data used by the package
│       │    ├── external    <- Data from third party sources.
│       │    ├── interim     <- Intermediate data that has been transformed.
│       │    ├── processed   <- The final, canonical data sets for modeling.
│       │    └── raw         <- The original, immutable data dump.
│       └── models           <- Trained and serialized models, model predictions,
│                               or model summaries.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.0rc2 and the [dsproject extension] 0.6.
For details and usage information on PyScaffold see <https://pyscaffold.org/>.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
