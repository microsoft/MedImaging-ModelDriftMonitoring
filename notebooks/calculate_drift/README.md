# Calculating Multi-Modal Concordance

## Notice
> This code is provided for research and development use only. This code is not intended for use in clinical decision-making or for any other clinical use and the performance of the code for clinical use has not been established. This source code requires selection of a reference and test datasets by the user. Microsoft does not warrant the suitability or accuracy of any predictive model generated using this source code. You bear sole responsibility for selection of a training dataset and for evaluation and use of any resulting model

## Prerequisites
Make sure you have a working AzureML Workspace, you have uploaded and created a dataset from the PadChest data, and uploaded our pretrained classifer and VAE (or trained your own).  If not, please complete our [setup tutorial](../setup/README.md).


## Generating VAE <img src="https://render.githubusercontent.com/render/math?math=Z"> and Predicted Probabilities
You will need to generate VAE <img src="https://render.githubusercontent.com/render/math?math=Z"> vectors and predicted probabilities for each of the images in the PadChest data.
Please follow these guides:
- Generating VAE $Z$: [Notebook](./generate_vae_data.ipynb)
- Generating predicted probabilities: [Notebook](./generate_model_score_data.ipynb.ipynb)

_Note_: Our VAE and predictions are available upon request.

## Calculate Individual Metrics:
Once you have <img src="https://render.githubusercontent.com/render/math?math=Z"> vectors and predicted probabilities you will need to calculate individual metrics using that data.
To this, you must create a result dataset in AzureML and to place the jsonl files, and PadChest metadata csv, then run a [script](../../src/scripts/drift/generate-drift-csv.py) to calculate metrics for PadChest.
Please follow this guide:
- Calculating individual drift metrics: [Notebook](./run-generate-drift.ipynb)

_Note_: Our individual metrics are provided as part of this release. 

## Unifying Metrics
After you have generated individual metrics, we use these to generate our unified metric, <img src="https://render.githubusercontent.com/render/math?math=\mathit{MMC}">. Please see this guide:
- Calculate and plot <img src="https://render.githubusercontent.com/render/math?math=\mathit{MMC}">: [Notebook](./calculate-mmc.ipynb)