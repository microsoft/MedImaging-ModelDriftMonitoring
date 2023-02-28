from pathlib import Path
from model_drift.data.dataset import MGBCXRDataset
from model_drift.data.datamodules import MGBCXRDataModule
import matplotlib.pyplot as plt
import numpy as np


image_dir = Path("/Volumes/MGB-CRE3-IRB2022P002646/")
csv_dir = Path("/Volumes/2015P002510/Chris/drift/data/csv")
# ds =  MGBCXRDataset(
#     dataframe_or_csv=csv_dir,
#     folder_dir=image_dir,
#     transform=lambda x: x,
#     frontal_only=True,
# )
# print("length", len(ds))

# for i in range(100):
#     el = ds[i]
#     im = np.array(el["image"])
#     # plt.imshow(im, cmap='gray')
#     # plt.show()
#     print(el["label"])
dm = MGBCXRDataModule(
    csv_folder=csv_dir,
    data_folder=image_dir,
    frontal_only=True,
    transforms=(lambda x: x)
)
dm.load_datasets()

el = dm.train_dataset[0]
im = np.array(el["image"])
plt.imshow(im, cmap='gray')
plt.show()
