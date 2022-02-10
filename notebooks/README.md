# Tutorial

Please start our tutorial [here](./setup/README.md).

## Notice
> This code is provided for research and development use only. This code is not intended for use in clinical decision-making or for any other clinical use and the performance of the code for clinical use has not been established. This source code requires selection of a reference and test datasets by the user. Microsoft does not warrant the suitability or accuracy of any predictive model generated using this source code. You bear sole responsibility for selection of a training dataset and for evaluation and use of any resulting model


## Individual Notebooks

- Setup [[Tutorial](./setup/README.md)]
- Train a customized VAE to detect image-based drift [[Notebook](setup/train_vae.ipynb)][[Script](src/scripts/vae/train.py)]
- Train a frontal-only model that is presumed to be deployed for further experiments [[Notebook](./setup/train_model.ipynb)][[Script](../src/scripts/finetune/train.py)]
- Generate VAE latent representations on PadChest [[Notebook](./calculate_drift/generate_vae_data.ipynb)][[Script](../src/scripts/vae/score.py)]
- Generate model predictions on PadChest [[Notebook](./calculate_drift/generate_model_score_data.ipynb)][[Script](../src/scripts/finetune/score.py)]
- Generate performance and individual metrics trials:
    - Unmodified PadChest data stream [[Notebook](./calculate_drift/run-generate-drift.ipynb)][[Script](../src/scripts/drift/generate-drift-csv.py)]
    - Performance degradation through hard data mining [[Notebook](./calculate_drift/run-generate-drift.ipynb)][[Script](../src/scripts/drift/generate-drift-csv.py)]
    - Clinical Workflow Failure through injecting lateral X-Ray images to a frontal-only model [[Notebook](./notebooks/calculate_drift/run-generate-drift.ipynb)][[Script](../src/scripts/drift/generate-drift-csv.py)]
    - Clinical Workflow Failure through injecting pediatric Chest X-Ray images to a frontal-only model [*Coming Soon*]
- Unification of statistical and model performance-based drift metrics [[Notebook](./calculate_drift/calculate-mmc.ipynb)]
- Explore VAE latent space [[Notebook](./appendix/explore-vae-latent-space.ipynb)]

