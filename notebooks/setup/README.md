# Setup

## Setup Azure ML Environment

1. Load PadChest into AzureML
2. Download pretrained models
3. Load into AzureML



## Steps

### Prerequisites:
AzureML Workspace: https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace

#### Compute resources:

`<TODO>`

### Load PadChest into Azure
Loading Data: https://github.com/microsoft/Medical-Imaging-AI-RSNA-2021-Demo/tree/main/1.Load%20Data

### Our pretrained models
- Download from [here](TBD)
Place in `<PROJECT_TOP>/models/`

### Loading models into AzureML

- Register AzureML Models [[Notebook](./train_model.ipynb)]

## Train your own models

 - Training VAE [[Notebook](./train_vae.ipynb)][[Script](../../src/scripts/vae/train.py)]
 - Training Classifier [[Notebook](./train_model.ipynb)][[Script](../../src/scripts/finetune/train.py)]