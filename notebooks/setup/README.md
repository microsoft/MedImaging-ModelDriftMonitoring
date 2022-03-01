# Setup Tutorial

_Note_: Though our tutorial includes the use of AzureML, our underlying scripts and libraries are built to work without AzureML.

## Notice
> This code is provided for research and development use only. This code is not intended for use in clinical decision-making or for any other clinical use and the performance of the code for clinical use has not been established. This source code requires selection of a reference and test datasets by the user. Microsoft does not warrant the suitability or accuracy of any predictive model generated using this source code. You bear sole responsibility for selection of a training dataset and for evaluation and use of any resulting model

## Setup AzureML Workspace
---
To run our experiments on AzureML, you will need to create an AzureML workspace, then add at least one compute cluster.
- Creating AzureML Workspace: [[Guide](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources#create-the-workspace)]
- Creating Compute Cluster: [[Guide](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster)]

## Compute Clusters
Our tutorial used two compute clusters, one for vae/model score generation, and another for metric calculation.
For vae/model score generation, we recommend a cluster VM that has 4 GPUs and 128+ GB of ram.
For metric calculation you may reuse the vae/model cluster or create another CPU only cluster. We recommend a CPU cluster with at least 24 cores and 64+ GB of ram.
These compute clusters are used throughout our tutorial.  Our GPU cluster is named `"nc24-uswest2"` and our CPU cluster is named `"cpu-cluster"`.  If you create clusters with different names, as you go through the tutorial, change these compute targets wherever you see them.

For reference, we used:

**GPU Cluster**
```
Name: Standard_NC24
Specs: 24 cores, 224GB RAM, 1440GB storage, 4 x NVIDIA Tesla K80
OS Type: Linux
Virtual machine tier: Dedicated
```

**CPU Cluster**

```
Name: Standard_D32_v3
Specs: 32 cores, 128 GB RAM, 800 GB disk
OS Type: Linux
Virtual machine tier: Dedicated
```

## Load PadChest into AzureML
To complete this tutorial you need to download and extract the PadChest dataset, then load it into AzureML.  
 - Loading PadChest: [[Guide](https://github.com/microsoft/Medical-Imaging-AI-RSNA-2021-Demo/tree/main/1.Load%20Data)]

## Download and Register Our Pretrained Models into AzureML

1. Place models[^1] in `<PROJECT_TOP>/models/`
2. Register AzureML models [[Notebook](./register_azureml_models.ipynb)]


[^1]: Available upon request.
## Train your own models (Optional)

 - Training VAE [[Notebook](./train_vae.ipynb)][[Script](../../src/scripts/vae/train.py)]
 - Training Classifier[^2] [[Notebook](./train_model.ipynb)][[Script](../../src/scripts/finetune/train.py)]
 - Register Your own AzureML Models [[Notebook](./register_azureml_models.ipynb)]

[^2]: Requires pretrained model, available upon request
# Whats Next?

After you complete these steps, continue the tutorial [here](../calculate_drift/README.md)


