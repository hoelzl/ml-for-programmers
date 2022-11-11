# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] tags=["slide"]
# <h1 style="text-align:center;">Machine Learning for Programmers</h1>
# <h2 style="text-align:center;">Setup (on Linux)</h2>
# <h3 style="text-align:center;">Dr. Matthias Hölzl</h3>

# %% [markdown] tags=["slide"]
# # Code for this Course
#
# `https://github.com/hoelzl/ml-for-programmers`
#
# ## Important Commits
#
# - `master` branch
# - `starter-kit-v1` tag

# %% [markdown] tags=["slide"]
# # Required Packages
#
# - numpy
# - pandas
# - matplotlib, seaborn
# - scikit-learn
# <hr/>
#
# - pytorch
# - fastai

# %% [markdown] tags=["slide"]
# <h1 style="text-align:center;">Setting up your environment</h1>
#

# %% [markdown] tags=["slide"]
# # Pip Install ...?
#
# - May or may not be what you want...
# - Use virtual environment(s)

# %% [markdown] tags=["slide"]
# # Hardware and OS
#
# - Traditional ML: OS does not matter
# - For Deep Learning: Linux, nVidia GPU
#   - Some libraries provide limited/no support for Windows
#   - Many DL models are much slower without GPU
#   - Only CUDA is well supported by all frameworks
#   - This may change over time (ROCm 4.0 is in beta for PyTorch 1.8)

# %% [markdown] tags=["subslide"]
# # If you use Windows
#
# - Cloud instances work well
# - Don't use a local VM
#   - Need to configure GPU passthrough
# - WSL2 works amazingly well...
#   - ... but right now only with the Insider Program Dev Channel

# %% [markdown] tags=["slide"]
# # Installation Options
#
# - `pip` + manual installation of libraries
# - `conda` (also installs native dependencies)
# - Mixed:
#   - `conda` for basics,
#   - `pip` for *everything else*

# %% [markdown] tags=["subslide"]
# # Not Recommended
#
# - `poetry` etc.
# - Truly mixed `conda` and `pip` install

# %% [markdown] tags=["slide"]
# # Installing Conda
#
# - Download Miniconda from <https://docs.conda.io/en/latest/miniconda.html#linux-installers>
# - Follow the instructions on <https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html>

# %% [markdown] tags=["slide"]
# # Setting up a Conda Environment
#
# - Download the code from GitHub to get the `environment.yml` file
# - Install using `conda env create --file environment.yml`
# - Don't update with `conda update --all` (or similar)
# - Use `conda env update --file environment.yml --prune` instead

# %% [markdown] tags=["slide"]
#
