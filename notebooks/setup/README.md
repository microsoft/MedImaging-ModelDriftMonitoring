# Setup

_Note_: Though our tutorial includes the use of AzureML, our underlying scripts and libraries are built to work without AzureML.

## Setup AzureML Workspace
To run our experiments on AzureML, you will need to create an AzureML workspace, then add at least one compute.  For compute clusters, we recommend a cluster VM that has 4 GPUs and 128+ GB of ram.
- AzureML Workspace: [[Guide](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace)]
- Creating Compute Cluster: [[Guide](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster)]

## Load PadChest into AzureML
To complete this tutorial you need to download and extract the PadChest dataset, then load it into AzureML.  
 - Loading PadChest: [[Guide](https://github.com/microsoft/Medical-Imaging-AI-RSNA-2021-Demo/tree/main/1.Load%20Data)]

## Download and Register Our Pretrained Models into AzureML

1. Download models [[Classifier](TBD), [VAE](TBD)]
2. Place in `<PROJECT_TOP>/models/`
3. Register AzureML models [[Notebook](./register_azureml_models.ipynb)]

## Train your own models (Optional)

 - Training VAE [[Notebook](./train_vae.ipynb)][[Script](../../src/scripts/vae/train.py)]
 - Training Classifier [[Notebook](./train_model.ipynb)][[Script](../../src/scripts/finetune/train.py)]
    - [CheXPert pretrained model](tbd)
 - Register Your own AzureML Models [[Notebook](./register_azureml_models.ipynb)]

