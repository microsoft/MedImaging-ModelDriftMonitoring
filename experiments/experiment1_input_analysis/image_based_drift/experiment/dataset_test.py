import argparse
import os
from dataset import PadChestDataset, ChestXrayDataset
from torch.utils.data import DataLoader
from azureml.core import Run, Model
from tqdm import tqdm
import torch

num_gpus = torch.cuda.device_count()
num_cpus = os.cpu_count()
node_rank = os.environ.get("NODE_RANK", 0)
local_rank = os.environ.get("LOCAL_RANK", 0)

print()
print("=" * 5)
# print(" Pytorch Lightning Version:", pl.__version__)
print(" Pytorch Version:", torch.__version__)
print(" Num GPUs:", num_gpus)
print(" Num CPUs:", num_cpus)
print(" Node Rank:", node_rank)
print(" Local Rank:", local_rank)
print("=" * 5)
print()

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data1", type=str, dest="data1", help="data folder mounting point"
)


parser.add_argument(
    "--data2", type=str, dest="data2", help="data folder mounting point"
)

parser.add_argument(
    "--batch_size", type=int, dest="batch_size", help="batch_size", default=64
)

parser.add_argument(
    "--image_size", type=int, dest="image_size", help="image_size", default=128
)

args = parser.parse_args()

data1 = PadChestDataset(
    args.data1,
    "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
    128,
    True,
    channels=1,
)

dataloader1 = DataLoader(
    dataset=data1,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_cpus,
    pin_memory=True,
)


data2 = ChestXrayDataset(
    args.data2,
    "train.csv",
    128,
    True,
    channels=1,
)

dataloader2 = DataLoader(
    dataset=data2,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=num_cpus,
    pin_memory=True,
)


run = Run.get_context()


step = 0
for batch1, batch2 in tqdm(
    zip(dataloader1, dataloader2), total=min(len(dataloader1), len(dataloader2))
):

    m1 = batch1["image"].mean()
    m2 = batch2["image"].mean()
    run.log("padchest/mean", float(m1), step=step)
    run.log("chexpert/mean", float(m2), step=step)

    run.log("padchest/max", float(batch1["image"].max()), step=step)
    run.log("chexpert/max", float(batch2["image"].max()), step=step)

    run.log("padchest/min", float(batch1["image"].min()), step=step)
    run.log("chexpert/min", float(batch2["image"].min()), step=step)

    run.log("padchest/mean.original", float(batch1["o_mean"].mean()), step=step)
    run.log("chexpert/mean.original", float(batch2["o_mean"].mean()), step=step)

    run.log("padchest/min.original", float(batch1["o_min"].min()), step=step)
    run.log("chexpert/min.original", float(batch2["o_min"].min()), step=step)

    run.log("padchest/max.original", float(batch1["o_max"].max()), step=step)
    run.log("chexpert/max.original", float(batch2["o_max"].max()), step=step)

    step += 1
