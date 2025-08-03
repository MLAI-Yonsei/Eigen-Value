# Eigen-Value: Method Implementation

This repository implements the method proposed in our paper, based on the [OpenDataVal](https://github.com/opendataval/opendataval) open-source package.

## Data Placement
- Place required datasets as follows:
    - CIFAR-10, VLCS: in the `/data` directory
    - CIFAR-10-C: in the `/data_files` directory
    - Amazon Reviews, ImageNet, DomainNet: Use embedding data located in the directory specified by the `--embedding_dir` argument when running the code

## Running Experiments
- Experiment scripts are located in `examples/CIFAR10` and `examples/VLCS`.
- After placing the data in the correct directories, run the relevant scripts in these folders to reproduce the results.

## Installation & Further Usage
- For installation instructions and additional usage details, please refer to the [original OpenDataVal GitHub repository](https://github.com/opendataval/opendataval).
- This repository only contains the minimal code required to implement our method; all core functionalities rely on OpenDataVal.

---

For questions or issues, please use the GitHub Issues page.
