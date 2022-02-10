> This code is provided for research and development use only. This code is not intended for use in clinical decision-making or for any other clinical use and the performance of the code for clinical use has not been established. This source code requires selection of a reference and test datasets by the user. Microsoft does not warrant the suitability or accuracy of any predictive model generated using this source code. You bear sole responsibility for selection of a training dataset and for evaluation and use of any resulting model


# Multi-Modal Drift Concordance For DL-based Automated Chest X-Ray Interpretation

<p align="left">
  <img alt="PyPI version" src="https://img.shields.io/pypi/v/libauc?color=blue&style=flat-square"/>
  <img alt="Python Version" src="https://img.shields.io/pypi/pyversions/libauc?color=blue&style=flat-square" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-1.8-yellow?color=blue&style=flat-square" />	
  <img alt="Tensorflow" src="https://img.shields.io/badge/Tensorflow-2.0-yellow?color=blue&style=flat-square" />
  <img alt="PyPI LICENSE" src="https://img.shields.io/github/license/yzhuoning/libauc?color=blue&logo=libauc&style=flat-square" />
</p>

[**Website**](https://libauc.org/)
| [**Updates**](https://libauc.org/news/)
| [**Installation**](https://libauc.org/get-started/)
| [**Tutorial**](https://github.com/Optimization-AI/LibAUC/tree/main/examples)
| [**Research**](https://libauc.org/publications/)
| [**Github**](https://github.com/Optimization-AI/LibAUC/)

MMC Drift aims to provide a system and set of experiments from the paper [CheXstray: Real-time Multi-Modal Data Concordance for Drift Detection in Medical Imaging AI](https://arxiv.org/abs/2202.02833) to test 
engineered dataset and model drift scenarios
that simulate different indicators for medical imaging AI drift in a production environment. If you use or like our library, please star⭐ our repo. Thank you!

Our framework has a
modular design and can be used in a plug-and-play manner to test multiple input drift modalities and scenarios with
include or new datasets.

## :mag: Why Multi-Modal Drift?

Rapidly expanding Clinical AI applications worldwide have the potential to impact to all areas of medical practice. Medical imaging applications constitute a vast majority of approved clinical AI applications.  A fundamental question remains: What happens after the AI model goes into production?

 We use the CheXpert and PadChest public datasets to build and test a medical imaging AI drift monitoring workflow that tracks data and model drift without contemporaneous ground truth. We simulate drift in multiple experiments to compare model performance with our novel multi-modal drift metric, which uses DICOM metadata, image appearance representation from a variational autoencoder (VAE), and model output probabilities as input. Through experimentation, we demonstrate a strong proxy for ground truth performance using unsupervised distributional shifts in relevant metadata, predicted probabilities, and VAE latent representation. 
 
 The key contributions of our research include: 
 
 (1) proof-of-concept for medical imaging drift detection including use of deep generative models (VAE) and domain specific statistical methods 
 
 (2) a multi-modal methodology for measuring and unifying drift metrics 
 
 (3) new insights into the challenges and solutions for observing deployed medical imaging AI 
 
 (4) creation of open-source tools (this repo) enabling others to easily run their own workflows or scenarios. This work has important implications for addressing the translation gap related to continuous medical imaging AI model monitoring in dynamic healthcare environments.



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
conda activate conda environment
conda develop src
```
    
:notebook_with_decorative_cover: Usage
-------
### Official Tutorials:

- Train a customized VAE to detect image-based drift [Notebook][Script]
- Train a frontal-only model that is presumed to be deployed for further experiments [Notebook][Script]
- Performance Degradation through hard data mining [Notebook][Script]
- Clinical Workflow Failure through injecting lateral X-Ray images to a frontal-only pipeline [Notebook][Script]
- Clinical Workflow Failure through injecting pediatric Chest X-Ray images to a frontal-only pipeline [Notebook][Script]
- Unification of statistical and model performance-based drift metrics [Notebook][Script]

## :zap: Useful Tips

- [ ]  Your Chest X-Ray dataset should have **0,1** labels, e.g., **1** represents **presence of finding** and **0** is the **absence of a given finding**
- [ ]  If there is no metadata or ground truth available, your drift analysis will be based on the Variational Autoencoder (VAE) and model output scores
## 🤝 Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.
## 📖 Citation

If you find MMC Drift useful in your work, please acknowledge our library and cite the following paper:

```
@misc{soin2022chexstray,
      title={CheXstray: Real-time Multi-Modal Data Concordance for Drift Detection in Medical Imaging AI}, 
      author={Arjun Soin and Jameson Merkow and Jin Long and Joesph Paul Cohen and Smitha Saligrama and Stephen Kaiser and Steven Borg and Ivan Tarapov and Matthew P Lungren},
      year={2022},
      eprint={2202.02833},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
## ✔️ Acknowledgements

 - [Microsoft Healthcare](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Stanford AIMI](https://github.com/matiassingers/awesome-readme)


## ©️ Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
