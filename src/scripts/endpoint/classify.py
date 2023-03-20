import os
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

# try:
#     import model_drift
# except ImportError:
#     library_path = str(Path(__file__).parent.parent.parent)
#     PYPATH = os.environ.get("PYTHONPATH", "").split(":")
#     if library_path not in PYPATH:
#         PYPATH.append(library_path)
#         os.environ["PYTHONPATH"] = ":".join(PYPATH)
#     import model_drift

# from model_drift import helpers
# from model_drift.models.finetune import CheXFinetune
# from model_drift.data.datamodules import PadChestDataModule, PediatricCheXpertDataModule, MIDRCDataModule
# from model_drift.callbacks import ClassifierPredictionWriter
# from model_drift.data.transform import VisionTransformer

def init():
    for name, value in os.environ.items():
        print("{0}: {1}".format(name, value))


@rawhttp
def run(request):
    pass