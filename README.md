
## FedIPSA: Personalized Federated Learning via Importance-Based Parameter Masking and Selective Alignment (paper in review - IEEE Internet of Things)


### Environment Preparation

#### PyPI 
```
pip install -r requirements.txt
```

### Run

#### Step 1. Generate FL Dataset
E.g. Partition the CIFAR-10 according to Dir(0.1) for 50 clients
```shell
python generate_data.py -d cifar10 -a 0.1 -cn 50
```

#### Step 2. Run Experiment

```sh
python main.py [--config-path, --config-name] [method=<METHOD_NAME> args...]
```

- `method`: The algorithm's name, e.g., `method=fedavg`. 
>   \[!NOTE\]
>   `method` should be identical to the `.py` file name in `src/server`.

- `--config-path`: Relative path to the directory of the config file. Defaults to `config`.
- `--config-name`: Name of `.yaml` config file (w/o the `.yaml` extension). Defaults to `defaults`, which points to `config/defaults.yaml`.

E.g. Running FedAvg with all default configurations. 
```sh
python main.py method=fedavg
```
Default configurations are set in both [`config/defaults.yaml`](config/defaults.yaml) and [`src/utils/constants.py`](src/utils/constants.py). See the folder ['config'](config/) for pre-set configurations that reproduce the experiments in the paper.

This repository is based on [FL-bench](https://github.com/KarhouTam/FL-bench/tree/master).
For more detail of the implementation and usage instructions, please refer to the original repository.


