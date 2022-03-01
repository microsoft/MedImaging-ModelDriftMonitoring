# Multi-Modal Drift Concordance For DL-based Automated Chest X-Ray Interpretation
## [**Paper**](https://arxiv.org/abs/2202.02833) | [**Issues**](https://github.com/microsoft/MedImaging-ModelDriftMonitoring/issues)


<p align="left">
  <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/libauc?color=blue&style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.8-yellow?color=blue&style=flat-square" />	
  <img alt="PyPI LICENSE" src="https://img.shields.io/badge/license-MIT-green" />
</p>

MMC Drift aims to provide a system and a set of experiments from the paper [CheXstray: Real-time Multi-Modal Data Concordance for Drift Detection in Medical Imaging AI](https://arxiv.org/abs/2202.02833) to test 
engineered dataset and model drift scenarios
that simulate different indicators for medical imaging AI drift in a production environment. If you use or like our library, please star⭐ our repo. Thank you!

Our framework has a
modular design and can be used in a plug-and-play manner to test multiple input drift modalities and scenarios that may include new datasets.

## :mag: Why Multi-Modal Drift?

Rapidly expanding Clinical AI applications worldwide have the potential to impact all areas of medical practice. Medical imaging applications constitute a vast majority of approved clinical AI applications.  A fundamental question remains: What happens after the AI model goes into production?

 We use the CheXpert and PadChest public datasets to build and test a medical imaging AI drift monitoring workflow that tracks data and model drift without contemporaneous ground truth. We simulate drift in multiple experiments to compare model performance with our novel multi-modal drift metric, which uses DICOM metadata, image appearance representation from a variational autoencoder (VAE), and model output probabilities as input. Through experimentation, we demonstrate a strong proxy for ground truth performance using unsupervised distributional shifts in relevant metadata, predicted probabilities, and VAE latent representation. 
 
 The key contributions of our research include: 
 
 (1) proof-of-concept for medical imaging drift detection including use of deep generative models (VAE) and domain specific statistical methods 
 
 (2) a multi-modal methodology for measuring and unifying drift metrics 
 
 (3) new insights into the challenges and solutions for observing deployed medical imaging AI 
 
 (4) creation of open-source tools (this repo) enabling others to easily run their own workflows or scenarios. This work has important implications for addressing the translation gap related to continuous medical imaging AI model monitoring in dynamic healthcare environments.


# Source Code

> This code is provided for research and development use only. This code is not intended for use in clinical decision-making or for any other clinical use and the performance of the code for clinical use has not been established. This source code requires selection of a reference and test datasets by the user. Microsoft does not warrant the suitability or accuracy of any predictive model generated using this source code. You bear sole responsibility for selection of a training dataset and for evaluation and use of any resulting model
## :star: Features

- Plug-and-play utilization
- Use PyTorch Lightning for quicker reproducibility
- Test engineered drift scenarios 
- Simulate both performance degradation and clinical workflow failure
- Gain insight into new methodologies (like drift metric unification) via ready-to-use Jupyter Notebooks
- Work with open-source or private datasets
- Hands-on AzureML Tutorials 


## :gear: Installation

Install the project and get up and running with conda

```bash
conda env create -f ./environment.yml --prefix ".venv"
conda activate "./.venv"
conda develop src
```
    
:notebook_with_decorative_cover: Usage
-------
- To test our MMC code using pre-obtained individual metrics, start [here](./notebooks/calculate_drift/calculate-mmc.ipynb)
- If you would like to train your own models, and generate your own metrics, start [here](./notebooks/setup/README.md).
- Links to individual notebooks are [here](./notebooks/README.md#individual-notebooks)
## :zap: Useful Tips

- [ ]  Your Chest X-Ray dataset should have **0,1** labels, e.g., **1** represents **presence of finding** and **0** is the **absence of a given finding**
- [ ]  If there is no metadata or ground truth available, your drift analysis will be based on the Variational Autoencoder (VAE) and model output scores
- [ ] Our tutorial uses AzureML, however our source code, scripts and notebooks are written such that they can used without AzureML!
# 🤝 Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a [pull request](https://github.com/microsoft/MedImaging-ModelDriftMonitoring/pulls), a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

These are the ways to contribute:
- [Submit bugs](https://github.com/microsoft/MedImaging-ModelDriftMonitoring/issues) and help us verify fixes as they are checked in.
- Contribute your own source code changes through a [pull request](https://github.com/microsoft/MedImaging-ModelDriftMonitoring/pulls).

_This project has adopted the Microsoft Open Source Code of Conduct. For more information see the [Code of Conduct FAQ](https://microsoft.github.io/codeofconduct/faq/) or contact opencode@microsoft.com with any additional questions or comments._


# :heart: Support

## How to file issues and get help  

This project uses GitHub [**issues**](https://github.com/microsoft/MedImaging-ModelDriftMonitoring/issues) to track bugs, feature requests, to ask for help and other questions about using this project. Please search the existing 
issues before filing new issues to avoid duplicates.  For new issues, file your bug or 
feature request as a new Issue.
# 📖 Citation

If you find MMC Drift useful in your work, please acknowledge our library and cite the following paper:

```
@misc{soin2022chexstray,
      title={CheXstray: Real-time Multi-Modal Data Concordance for Drift Detection in Medical Imaging AI}, 
      author={Arjun Soin and Jameson Merkow and Jin Long and Joseph Paul Cohen and Smitha Saligrama and Stephen Kaiser and Steven Borg and Ivan Tarapov and Matthew P Lungren},
      year={2022},
      eprint={2202.02833},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
# ✔️ Acknowledgements

 - [Microsoft Healthcare](https://cloudblogs.microsoft.com/industry-blog/health/)
 - [Stanford AIMI](https://aimi.stanford.edu/)

# :eyes: Extras

Check out these other projects from Microsoft!


## InnerEye - Advanced Machine Learning algorithms for medical imaging
If you are looking for advanced algorithms for medical image analysis that harness the power of Azure-based cloud computing - make sure to check out the amazing work done by the InnerEye team from Microsoft Research Cambridge and their open-source repositories: 
* [InnerEye-DeepLearning](https://github.com/microsoft/InnerEye-DeepLearning) for advanced medical image analysis algorithms 
* [HI-ML](https://github.com/microsoft/hi-ml) - building blocks for AI/ML scenarios for medical applications: 

## DICOM Service
A real world medical imaging AI setup would leverage DICOM data and would benefit from storing DICOM images in the cloud. You can explore the services that Microsoft provides to manage DICOM and FHIR data in the cloud: 
* [DICOM service](https://docs.microsoft.com/en-us/azure/healthcare-apis/dicom/dicom-services-overview)
* [FHIR service](https://docs.microsoft.com/en-us/azure/healthcare-apis/fhir/overview)


<!-- * [Microsoft Healthcare](https://www.microsoft.com/en-us/industry/health/microsoft-cloud-for-healthcare) -->
## Microsoft Medical Imaging AI RSNA 2021 Demo
This demo will walk you through the steps that a data science team would typically undertake and describe the use of Microsoft tools as well as provide some custom code to take you through the steps. The focus of this demo is not on the algorithms used to build the best performing system, but rather the steps and tools that are needed to get one there. The same tools and principles could be applied to many other types of medical imaging datasets. 
* [GitHub](https://github.com/microsoft/Medical-Imaging-AI-RSNA-2021-Demo)

## Medical Imaging AI Scenarios
TBD: Andreas' demo

## Learning materials
Here are some learning materials if you would like to explore some of the Microsoft's AI tools further: 
* [A 30 day challenge](https://docs.microsoft.com/en-us/learn/challenges?id=8E1F62A7-99E3-48E4-9EC9-1FFFB99EE9AF&wt.mc_id=cloudskillschallenge_8E1F62A7-99E3-48E4-9EC9-1FFFB99EE9AF)  focusing on learning AI fundamentals
* [Interactive introduction into Azure ML](https://docs.microsoft.com/en-us/learn/modules/intro-to-azure-ml/)

# ©️ Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
