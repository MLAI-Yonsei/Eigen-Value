# Eigen-Value: Method Implementation

This repository implements the method proposed in our paper, based on the [OpenDataVal](https://github.com/opendataval/opendataval) open-source package.



## Installation & Further Usage
- For installation instructions and additional usage details, please refer to the [original OpenDataVal GitHub repository](https://github.com/opendataval/opendataval).
- This repository only contains the minimal code required to implement our method; all core functionalities rely on OpenDataVal.


## Data Placement
- Place required datasets as follows:
    - CIFAR-10, VLCS: in the `/data` directory
    - CIFAR-10-C: in the `/data_files` directory
    - Amazon Reviews, ImageNet, DomainNet: Use embedding data located in the directory specified by the `--embedding_dir` argument when running the code

## Running Experiments
- Experiment scripts are located in `examples/CIFAR10` and `examples/VLCS`.
- After placing the data in the correct directories, run the relevant scripts in these folders to reproduce the results.

### Data Valuation
To begin data valuation for each sample, run the following code.
This script computes the data value for each sample according to the chosen data valuation method, splits them by data type, and saves the resulting values into CSV files.
```bash
python ./examples/CIFAR10/datavaluation_CIFAR10embedding.py
```
### Data Removal
Using the data value files generated in the Data Valuation step, the data removal experiment is conducted.
Based on the computed data values, the top 50% (highest-value samples) are removed, and a logistic regression model is trained only on the remaining data.
```bash
python ./examples/CIFAR10/removal.py --ascending True --num 500
```
### Point Addition
Similarly, using the data values computed during Data Valuation, the point addition experiment is performed.
By adjusting the `num` argument, you can choose how many top-valued samples to add to a fixed dataset.
```bash
python ./examples/CIFAR10/removal.py --ascending false --num 100
```

### Instability Ranking
This experiment was conducted only on CIFAR-10.
During Data Valuation, a specific subset is fixed while other parts of the dataset are randomly varied.
The goal is to measure how the data values of the fixed subset change under different conditions.
The same code is executed multiple times with different seeds (for different parts of dataset are sampled).
```bash
python ./examples/CIFAR10/instability_CIFAR10embedding.py --ver_ID --data_i 10
```

## Setting


We used a fixed random seed of 42 for all experiments (except for the Instability Ranking experiment).
As described in the paper, the embedding models used were ResNet-50 and ViT-B/16, and we trained a logistic regression model on top of the embeddings.
All training data came from the train split of each dataset.

For CIFAR-10, we computed data valuation using the train split. Each experiment was then performed by training on that split and evaluating performance on CIFAR-10-C.
Similarly, for VLCS, data values were computed using the train split of all domains except the target domain. After training on those domains, performance was evaluated on the train split of the target domain.


---

For questions or issues, please use the GitHub Issues page.
