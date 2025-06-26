# QPyTorch

---
[![Test Suite](https://github.com/lanzithinking/qpytorch/actions/workflows/run_test_suite.yml/badge.svg)](https://github.com/lanzithinking/qpytorch/actions/workflows/run_test_suite.yml)
[![Documentation Status](https://readthedocs.org/projects/qpytorch/badge/?version=stable)](https://qpytorch.readthedocs.io/en/stable/?badge=stable)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Conda](https://img.shields.io/conda/v/qpytorch/qpytorch.svg)](https://anaconda.org/qpytorch/qpytorch)
[![PyPI](https://img.shields.io/pypi/v/qpytorch.svg)](https://pypi.org/project/qpytorch)

QPyTorch is a Q-exponential process library implemented using PyTorch built on [GPyTorch](https://gpytorch.ai). QPyTorch is designed for creating scalable, flexible, and modular Q-exponential process models with ease.

Internally, QPyTorch differs from many existing approaches to QEP inference by performing most inference operations using numerical linear algebra techniques like preconditioned conjugate gradients.
Implementing a scalable QEP method is as simple as providing a matrix multiplication routine with the kernel matrix and its derivative via our [LinearOperator](https://github.com/cornellius-gp/linear_operator) interface,
or by composing many of our already existing `LinearOperators`.
This allows not only for easy implementation of popular scalable QEP techniques,
but often also for significantly improved utilization of GPU computing compared to solvers based on the Cholesky decomposition.

QPyTorch provides (1) significant GPU acceleration (through MVM based inference);
(2) state-of-the-art implementations of the latest algorithmic advances for scalability and flexibility ([SKI/KISS-GP](http://proceedings.mlr.press/v37/wilson15.pdf), [stochastic Lanczos expansions](https://arxiv.org/abs/1711.03481), [LOVE](https://arxiv.org/pdf/1803.06058.pdf), [SKIP](https://arxiv.org/pdf/1802.08903.pdf), [stochastic variational](https://arxiv.org/pdf/1611.00336.pdf) [deep kernel learning](http://proceedings.mlr.press/v51/wilson16.pdf), ...);
(3) easy integration with deep learning frameworks.


## Examples, Tutorials, and Documentation

See our [**documentation, examples, tutorials**](https://qpytorch.readthedocs.io/en/stable/) on how to construct all sorts of models in QPyTorch.

## Installation

**Requirements**:
- Python >= 3.8
- PyTorch >= 2.0
- GPyTorch >= 1.13

Install QPyTorch using pip or conda:

```bash
pip install qpytorch
conda install qpytorch -c qpytorch
```

(To use packages globally but install QPyTorch as a user-only package, use `pip install --user` above.)

#### Latest (Unstable) Version

To upgrade to the latest (unstable) version, run

```bash
pip install --upgrade git+https://github.com/cornellius-gp/linear_operator.git
pip install --upgrade git+https://github.com/cornellius-gp/gpytorch.git
pip install --upgrade git+https://github.com/lanzithinking/qpytorch.git
```

#### Development version

If you are contributing a pull request, it is best to perform a manual installation:

```sh
git clone https://github.com/lanzithinking/qpytorch.git
cd qpytorch
pip install -e .[dev,docs,examples,keops,pyro,test]  # keops and pyro are optional
```

#### ArchLinux Package
**Note**: Experimental AUR package. For most users, we recommend installation by conda or pip.

QPyTorch is also available on the [ArchLinux User Repository](https://wiki.archlinux.org/index.php/Arch_User_Repository) (AUR).
You can install it with an [AUR helper](https://wiki.archlinux.org/index.php/AUR_helpers), like [`yay`](https://aur.archlinux.org/packages/yay/), as follows:

```bash
yay -S python-qpytorch
```
To discuss any issues related to this AUR package refer to the comments section of
[`python-qpytorch`](https://aur.archlinux.org/packages/python-qpytorch/).


## Citing Us

If you use QPyTorch, please cite the following papers:
> [Li, Shuyi, Michael O'Connor, and Shiwei Lan. "Bayesian Learning via Q-Exponential Process." In Advances in Neural Information Processing Systems (2023).](https://papers.nips.cc/paper_files/paper/2023/hash/e6bfdd58f1326ff821a1b92743963bdf-Abstract-Conference.html)
```
@inproceedings{li2023QEP,
  title={Bayesian Learning via Q-Exponential Process},
  author={Li, Shuyi, Michael O'Connor, and Shiwei Lan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```

## Contributing

See the contributing guidelines [CONTRIBUTING.md](https://github.com/lanzithinking/qpytorch/blob/main/CONTRIBUTING.md)
for information on submitting issues and pull requests.


## The Team

QPyTorch is primarily maintained by:
- [Shiwei Lan](https://math.la.asu.edu/~slan) (Arizona State University)

We would like to thank our other contributors including (but not limited to)
Shuyi Li,
Guangting Yu,
Zhi Chang,
Chukwudi Paul Obite,
Keyan Wu,
and many more!


## Acknowledgements
Development of QPyTorch is supported by.


## License

QPyTorch is [MIT licensed](https://github.com/lanzithinking/qpytorch/blob/main/LICENSE).
