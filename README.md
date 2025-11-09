# Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch <img src="QePyTorch.PNG" alt="Logo" style="height: 1em; vertical-align: middle;">

---
[![Test Suite](https://github.com/lanzithinking/qepytorch/actions/workflows/run_test_suite.yml/badge.svg)](https://github.com/lanzithinking/qepytorch/actions/workflows/run_test_suite.yml)
[![Documentation Status](https://readthedocs.org/projects/qepytorch/badge/?version=latest)](https://qepytorch.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Conda](https://img.shields.io/conda/v/conda-forge/qpytorch.svg)](https://anaconda.org/conda-forge/qpytorch)
[![PyPI](https://img.shields.io/pypi/v/qpytorch.svg)](https://pypi.org/project/qpytorch)

Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch is a Python package for Q-exponential process ([QEP](https://papers.nips.cc/paper_files/paper/2023/file/e6bfdd58f1326ff821a1b92743963bdf-Paper-Conference.pdf)) implemented using PyTorch and built upon [GPyTorch](https://gpytorch.ai). Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch is designed to facilitate creating scalable, flexible, and modular QPE models.

Different from GPyTorch for Gaussian process (GP) models, Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch focuses on QEP, which generalizes GP by allowing flexible regularization on function spaces through a parameter $q>0$ and embraces GP as a special case with $q=2$. QEP is proven to be superior than GP in modeling inhomogeneous objects with abrupt changes or sharp contrast for $q<2$ [[Li et al (2023)]](https://papers.nips.cc/paper_files/paper/2023/hash/e6bfdd58f1326ff821a1b92743963bdf-Abstract-Conference.html).
Inherited from GPyTorch, Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch has an efficient and scalable implementation by taking advantage of numerical linear algebra library [LinearOperator](https://github.com/cornellius-gp/linear_operator) and improved GPU utilization.


<!--
Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch features ...
-->


## Tutorials, Examples, and Documentation

See [**documentation**](https://qepytorch.readthedocs.io/en/stable/) on how to construct various QEP models in Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch.

## Installation

**Requirements**:
- Python >= 3.10
- PyTorch >= 2.0
- GPyTorch >= 1.14

#### Stable Version

Install Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch using pip or conda:

```bash
pip install qpytorch
conda install qpytorch
```

(To use packages globally but install Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch as a user-only package, use `pip install --user` above.)

#### Latest Version

To upgrade to the latest version, run

```bash
pip install --upgrade git+https://github.com/lanzithinking/qepytorch.git
```

#### from source (for development)

If you are contributing a pull request, it is best to perform a manual installation:

```sh
git clone https://github.com/lanzithinking/qepytorch.git
cd qepytorch
# either
pip install -e .[dev,docs,examples,keops,pyro,test]  # keops and pyro are optional
# or
conda env create -f env_install.yaml # installed in the environment qpytorch
```

<!--
#### ArchLinux Package
**Note**: Experimental AUR package. For most users, we recommend installation by conda or pip.
-->
<!--
Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch is also available on the [ArchLinux User Repository](https://wiki.archlinux.org/index.php/Arch_User_Repository) (AUR).
You can install it with an [AUR helper](https://wiki.archlinux.org/index.php/AUR_helpers), like [`yay`](https://aur.archlinux.org/packages/yay/), as follows:
-->
<!--
```bash
yay -S python-qpytorch
```
To discuss any issues related to this AUR package refer to the comments section of
[`python-qpytorch`](https://aur.archlinux.org/packages/python-qpytorch/).
-->

## Citing Us

If you use Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch, please cite the following paper:
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

See the contributing guidelines [CONTRIBUTING.md](https://github.com/lanzithinking/qepytorch/blob/main/CONTRIBUTING.md)
for information on submitting issues and pull requests.


## The Team

Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch is primarily maintained by:
- [Shiwei Lan](https://math.la.asu.edu/~slan) (Arizona State University)

Thanks to the following contributors including (but not limited to)
- Shuyi Li,
Guangting Yu,
Zhi Chang,
Chukwudi Paul Obite,
Keyan Wu,
and many more!

<!--
## Acknowledgements
Development of Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch is supported by.
-->

## License

Q<sup style="font-size: 0.5em;">&#9428;</sup>PyTorch is [MIT licensed](https://github.com/lanzithinking/qepytorch/blob/main/LICENSE).
