# 696_experiments
 
### Data files
`dataset_synthesis.ipynb` is the notebook for dataset synthesis. Following are the datafiles
1. `data_unif.csv` - Data points for all clusters sampled from a uniform distribution
2. `data_norm.csv` - Data points for all clusters sampled from a normal distribution with spherical covariance

### Running code
Do `python main.py`
For wandb and hyperparameter tuning check `main_with_sweep.py` and `sweep.yaml`

### Modifications
1. `src/model.py` contains the model definition
2. `src/runner.py` contains the train and test code
3. `src/viz.py` contains the the visualization code
4. `src/data.py` contains the data construct
5. `main.py` is the wrapper for the experiments
