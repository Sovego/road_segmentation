#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from multiprocessing import freeze_support
from os.path import join as pjoin

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchmetrics.classification as metrics

import torchvision
from torchvision import transforms

import segmentation_models_pytorch as smp

from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import torchinfo

import matplotlib.pyplot as plt

# In[2]:


import os
from os.path import join as pjoin
import contextlib

import cv2
import numpy as np

import matplotlib.pyplot as plt


def uniqufy_path(path):
    filename, extension = os.path.splitext(path)
    file_index = 1

    while os.path.exists(path):
        path = f"{filename}_{file_index}{extension}"
        file_index += 1

    return path


def create_image_plot(row_len: int = None, figsize=(16, 6), **images):
    n_images = len(images)
    if row_len is None:
        row_len = n_images
    fig = plt.figure(figsize=figsize)
    for idx, (name, image) in enumerate(images.items()):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(n_images // row_len + 1, row_len, idx + 1)
        ax.set_title(name.title(), fontsize=16)
        with open("$null", 'w') as dummy_f:
            with contextlib.redirect_stderr(dummy_f):
                ax.imshow(image)
    return fig


def save_imgs(path=None, name="imgs", **images):
    if (path is None):
        raise AttributeError(f"You shoud write path")
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = pjoin(path, f"{name}")
    fig = create_image_plot(**images)
    fig.savefig(image_path)
    fig.clear()
    plt.close(fig)


# In[3]:


LAUNCH_NAME = "UNet_10"

STARTING_EPOCH = 0
LOAD_WEIGHTS = None  #
LOAD_ADAM_STATE = None  #
USE_MANUAL_TENSORBOARD_FOLDER = None  #

SAVED_MODEL_PATH = None

EPOCHS = 40
LEARNING_RATE = 1E-5  # 0.0001
WEIGHT_DECAY = 0  # 1E-7

BATCH_SIZE = 10  # 20

SAVE_METHOD = "TORCH"  # "TORCH" / "ONNX"
WEIGHT_SAVER = "last"  # "all" / "nothing" / "last"

CLASS_NAMES = ['other', 'road']
CLASS_RGB_VALUES = [[0, 0, 0], [255, 255, 255]]

NORMALIZE_MEAN_IMG = [0.4295, 0.4325, 0.3961]  # [0.485, 0.456, 0.406]
NORMALIZE_DEVIATIONS_IMG = [0.2267, 0.2192, 0.2240]  # [0.229, 0.224, 0.225]

CROP_SIZE = (256, 256)

NUM_WORKERS = 4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATASET_DIR = './tiff'
VALID_SET = (pjoin(DATASET_DIR, "val"), pjoin(DATASET_DIR, "val_labels"))
TEST_SET = (pjoin(DATASET_DIR, "test"), pjoin(DATASET_DIR, "test_labels"))
TRAIN_SET = (pjoin(DATASET_DIR, "train"), pjoin(DATASET_DIR, "train_labels"))

trained = False

# In[4]:


TBpath = uniqufy_path(
    f"TB_cache/{LAUNCH_NAME}") if USE_MANUAL_TENSORBOARD_FOLDER is None else USE_MANUAL_TENSORBOARD_FOLDER
TBwriter = SummaryWriter(TBpath)


# In[5]:


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


prepare_to_network = A.Lambda(image=to_tensor, mask=to_tensor)

train_transform = A.Compose(
    [
        A.OneOf(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
        A.Normalize(mean=NORMALIZE_MEAN_IMG, std=NORMALIZE_DEVIATIONS_IMG, always_apply=True)
    ]
)

valid_transform = A.Compose(
    [
        A.Normalize(mean=NORMALIZE_MEAN_IMG, std=NORMALIZE_DEVIATIONS_IMG, always_apply=True),
    ]
)


# In[6]:


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x


# In[7]:


class RoadsDataset(Dataset):
    def __init__(self, values_dir, labels_dir, class_rgb_values=None, transform=None, readyToNetwork=None):
        self.values_dir = values_dir
        self.labels_dir = labels_dir
        self.class_rgb_values = class_rgb_values
        self.images = [pjoin(self.values_dir, filename) for filename in sorted(os.listdir(self.values_dir))]
        self.labels = [pjoin(self.labels_dir, filename) for filename in sorted(os.listdir(self.labels_dir))]
        self.transform = transform
        self.readyToNetwork = readyToNetwork

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.labels[index]

        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        label = one_hot_encode(label, self.class_rgb_values).astype('float')

        if self.transform:
            sample = self.transform(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        if self.readyToNetwork:
            sample = self.readyToNetwork(image=image, mask=label)
            image, label = sample['image'], sample['mask']
        return image, label


# In[8]:


sample_dataset = RoadsDataset(*TEST_SET,
                              class_rgb_values=CLASS_RGB_VALUES, transform=valid_transform)

for i in range(10):
    image, mask = sample_dataset[np.random.randint(0, len(sample_dataset))]
    TBwriter.add_figure(f'train samples', create_image_plot(origin=image, true=colour_code_segmentation(
        reverse_one_hot(mask), CLASS_RGB_VALUES)), global_step=i)
del (sample_dataset)

# In[9]:


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# In[10]:


""" Full assembly of the parts to form the complete network """


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# In[11]:


model = UNet(3, 2, bilinear=True).to(DEVICE)

loss = smp.losses.DiceLoss(mode='binary')

optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, threshold=1e-3, cooldown=1, factor=0.5)

# In[12]:


train_dataset = RoadsDataset(*TRAIN_SET,
                             class_rgb_values=CLASS_RGB_VALUES, transform=train_transform,
                             readyToNetwork=prepare_to_network)
valid_dataset = RoadsDataset(*VALID_SET,
                             class_rgb_values=CLASS_RGB_VALUES, transform=valid_transform,
                             readyToNetwork=prepare_to_network)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE // 4,
    num_workers=NUM_WORKERS,
)


# In[13]:


# images, _ = next(iter(valid_dataloader))
# TBwriter.add_graph(model, images)


# In[14]:


# In[15]:


def train_step(net, criterion, optimizer, dataloader, epoch: int = None):
    net.train()
    running_loss = 0.
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss

    with torch.no_grad():
        train_loss = running_loss / len(dataloader)
    return train_loss.item()


def valid_step(net, criterion, dataloader, epoch: int = None):
    net.eval()
    running_loss = 0.
    IoU = metrics.BinaryJaccardIndex()
    IoU.to(DEVICE)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            output = net(images)

            IoU(output, labels)
            loss = criterion(output, labels)
            running_loss += loss

            save_imgs(pjoin(TBpath, f"valid_samples/samples_{epoch}"), name=f"img_{step}",
                      origin=images[0].cpu().numpy().transpose(2, 1, 0),
                      true=colour_code_segmentation(reverse_one_hot(
                          labels[0].cpu().numpy().transpose(2, 1, 0)), CLASS_RGB_VALUES),
                      pred=colour_code_segmentation(reverse_one_hot(
                          output[0].cpu().numpy().transpose(2, 1, 0)), CLASS_RGB_VALUES))

        TBwriter.add_figure('valid_sample', create_image_plot(
            origin=images[0].cpu().numpy().transpose(2, 1, 0),
            true=colour_code_segmentation(reverse_one_hot(
                labels[0].cpu().numpy().transpose(2, 1, 0)), CLASS_RGB_VALUES),
            pred=colour_code_segmentation(reverse_one_hot(
                output[0].cpu().numpy().transpose(2, 1, 0)), CLASS_RGB_VALUES)),
                            epoch)

        valid_loss = running_loss / len(valid_dataloader)

        return valid_loss.item(), IoU.compute().item()


# In[16]:


# In[ ]:


# In[ ]:


def test_step(model, loader):
    classes = CLASS_NAMES

    iou = metrics.JaccardIndex(task="multiclass", num_classes=2).to(DEVICE)

    with torch.no_grad():
        model.eval()
        for id, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(images)
            TBwriter.add_figure('test_sample', create_image_plot(
                origin=images[0].cpu().numpy().transpose(2, 1, 0),
                true=colour_code_segmentation(reverse_one_hot(
                    labels[0].cpu().numpy().transpose(2, 1, 0)), CLASS_RGB_VALUES),
                pred=colour_code_segmentation(reverse_one_hot(
                    output[0].cpu().numpy().transpose(2, 1, 0)), CLASS_RGB_VALUES)),
                                id)
            iou.update(output, labels)
    return iou.compute().cpu()


# In[ ]:


# In[ ]:

if __name__ == '__main__':
    freeze_support()
    print(model_sum := torchinfo.summary(model, input_size=(BATCH_SIZE, 3, *CROP_SIZE), row_settings=["var_names"],
                                         verbose=0, col_names=[
            "input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"]))
    epoch = STARTING_EPOCH

    # In[ ]:

    best_loss = 10000
    trained = True

    pbar = tqdm(range(EPOCHS))
    pbar.update(epoch)

    while (epoch < EPOCHS):
        train_loss = train_step(model, loss, optimizer, train_dataloader, epoch)
        valid_loss, iou_score = valid_step(model, loss, valid_dataloader, epoch)
        scheduler.step(valid_loss)

        if WEIGHT_SAVER != "nothing" and valid_loss < best_loss and epoch > 3:
            best_loss = valid_loss

            print(f"[{epoch}] Saved weights with IoU: {iou_score:.2f} | loss: {valid_loss:.4f}")

            if WEIGHT_SAVER == "all":
                weights_path = f"{TBpath}/weights_{epoch}.pth"
                model_path = f"{TBpath}/model_{epoch}.onnx"
                optimizer_path = f"{TBpath}/optimizer_{epoch}.pth"

            elif WEIGHT_SAVER == "last":
                weights_path = f"{TBpath}/weights_last.pth"
                model_path = f"{TBpath}/model_last.onnx"
                optimizer_path = f"{TBpath}/optimizer_last.pth"

            if "TORCH" in SAVE_METHOD:
                torch.save(model.state_dict(), weights_path)

            if "ONNX" in SAVE_METHOD:
                torch.onnx.export(model, torch.empty(size=(BATCH_SIZE, 3, *CROP_SIZE)), model_path)

        TBwriter.add_scalar('valid loss', valid_loss, epoch)
        TBwriter.add_scalar('train loss', train_loss, epoch)

        TBwriter.add_scalar('IoU', iou_score, epoch)

        epoch += 1
        pbar.update()
        pbar.set_description(
            f'IoU: {iou_score:.2f}  | train/valid loss: {train_loss:.4f}/{valid_loss:.4f}')
    test_transform = A.Compose(
        [
            A.Normalize(mean=NORMALIZE_MEAN_IMG, std=NORMALIZE_DEVIATIONS_IMG, always_apply=True),
        ]
    )

    test_dataset = RoadsDataset(*TEST_SET,
                                class_rgb_values=CLASS_RGB_VALUES, transform=valid_transform,
                                readyToNetwork=prepare_to_network)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=36,
        num_workers=NUM_WORKERS,
    )
    if not trained:
        print(f"Используется не обученная модель, происходит загрузка модели из {SAVED_MODEL_PATH}")
        model = None
        if "ONNX" in SAVE_METHOD and model is None:
            print(f"Попытка импорта модели из onnx файла")
            try:
                import onnx

                model = onnx.load(SAVED_MODEL_PATH)
            except:
                pass
        if "TORCH" in SAVE_METHOD and model is None:
            print(f"Попытка импорта модели из pth файла")
            model = UNet(3, 2, bilinear=True)
            model.state_dict(torch.load(f=SAVED_MODEL_PATH))

        model.to(DEVICE)

    iou = test_step(model, test_dataloader)
    print(f"IoU: {iou}")
    TBwriter.close()
    # In[ ]:
