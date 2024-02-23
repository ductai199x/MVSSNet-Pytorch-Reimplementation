import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchvision.transforms import Compose, Normalize, Resize
from torchvision.transforms.functional import resize
from wandb.sdk.wandb_run import Run

from data.videofact2_dataset import VideoFact2Dataset

from model.mvss import get_mvss

normalize_dict = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def collate_fn(batch):
    frames, label, masks, initial_shape = tuple(zip(*batch))
    frames = torch.stack(frames)
    label = torch.tensor(label)
    masks = torch.stack(masks)
    initial_shape = torch.stack(initial_shape)
    return frames, label, masks, initial_shape


class MVSS_PLWrapper(LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.model_config = model_config
        self.training_config = training_config
        self.model = get_mvss(
            backbone="resnet50",
            pretrained_base=True,
            nclass=1,
            sobel=True,
            constrain=True,
            n_input=3,
        )

        self.train_det_acc = BinaryAccuracy()
        self.train_loc_f1 = BinaryF1Score()
        self.val_det_acc = BinaryAccuracy()
        self.val_loc_f1 = BinaryF1Score()

        self.transforms = Compose(
            [
                Resize((512, 512), antialias=True),
                Normalize(**normalize_dict),
            ]
        )

        self.save_hyperparameters()

    def preprocess(self, x):
        return self.transforms(x)

    def dice_loss(self, x, y):
        true_pos = torch.sum(x * y)
        false_pos = torch.sum((1 - x) * y)
        false_neg = torch.sum(x * (1 - y))
        dice_loss = 1 - (2 * true_pos) / (2 * true_pos + false_pos + false_neg)
        return dice_loss

    def calc_loss(self, pred_class, edge_mask, pred_mask, class_label, gt_mask):
        alpha, beta = self.model_config["alpha"], self.model_config["beta"]
        gt_edge_mask = resize(gt_mask, size=(edge_mask.shape[-2], edge_mask.shape[-1]), antialias=False)
        det_loss = F.binary_cross_entropy(pred_class, class_label.float())
        loc_loss = self.dice_loss(pred_mask, gt_mask)
        edge_loc_loss = self.dice_loss(edge_mask, gt_edge_mask)
        loss = alpha * loc_loss + beta * det_loss + (1 - alpha - beta) * edge_loc_loss
        return loss

    def forward(self, x):
        x = self.preprocess(x)
        edge_mask, pred_mask =  self.model(x)
        pred_mask = torch.sigmoid(pred_mask)
        edge_mask = torch.sigmoid(edge_mask)
        pred_class = pred_mask.flatten(1, -1).max(1).values
        return pred_class, edge_mask, pred_mask

    @torch.no_grad()
    def log_loc_output(self, x, gt_mask, pred_mask, step_idx):
        x = x.detach().cpu()
        gt_mask = gt_mask.float().detach().cpu()
        pred_mask = pred_mask.float().detach().cpu()
        logger = self.logger.experiment
        if isinstance(logger, Run):
            log_images = []
            log_images.append(wandb.Image(x, caption="input"))
            log_images.append(wandb.Image(gt_mask, caption="gt_mask"))
            log_images.append(wandb.Image(pred_mask, caption="pred_mask"))
            logger.log({"train_loc_output": log_images}, step=step_idx)
        elif isinstance(logger, SummaryWriter):
            logger.add_images("train_loc_x", x, dataformats="CHW", global_step=step_idx)
            logger.add_images("train_loc_gt", gt_mask, dataformats="HW", global_step=step_idx)
            logger.add_images("train_loc_pred", pred_mask, dataformats="HW", global_step=step_idx)
        else:
            pass

    def training_step(self, batch, batch_idx):
        x, y, m, s = batch
        pred_class, edge_mask, pred_mask = self.forward(x)
        pred_mask = resize(pred_mask, size=(m.shape[-2], m.shape[-1]), antialias=False).squeeze(1)
        edge_mask = edge_mask.squeeze(1)
        loss = self.calc_loss(pred_class, edge_mask, pred_mask, y, m)

        self.train_det_acc(pred_class, y)
        self.train_loc_f1(pred_mask, m)

        if self.global_step % 200 == 0:
            self.log_loc_output(x[-1], m[-1], pred_mask[-1], self.global_step)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_det_acc", self.train_det_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loc_f1", self.train_loc_f1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, m, s = batch
        x = self.preprocess(x)
        pred_class, edge_mask, pred_mask = self.forward(x)
        pred_mask = resize(pred_mask, size=(m.shape[-2], m.shape[-1]), antialias=False).squeeze(1)
        edge_mask = edge_mask.squeeze(1)
        loss = self.calc_loss(pred_class, edge_mask, pred_mask, y, m)

        self.val_det_acc(pred_class, y)
        self.val_loc_f1(pred_mask, m)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_det_acc", self.val_det_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loc_f1", self.val_loc_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.training_config["lr"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.training_config["decay_step"], gamma=self.training_config["decay_rate"]
        )
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        ds_root_dir = "/media/nas2/videofact2_data_large"
        ds_metadata_path = f"{ds_root_dir}/metadata.csv"
        metadata = pd.read_csv(ds_metadata_path)
        metadata = metadata.query("split == 'train'")
        manip_types = self.training_config["manip_types"]
        metadata_ = []
        for manip_type in manip_types:
            manip_metadata = metadata.query(f"vid_id.str.contains('{manip_type}')")
            manip_length = len(manip_metadata)
            if isinstance(manip_types[manip_type], int):
                n = manip_types[manip_type]
                frac = None
                replace = True if n > manip_length else False
            else:
                n = None
                frac = manip_types[manip_type]
                replace = True if frac > 1 else False
            manip_metadata = manip_metadata.sample(n=n, frac=frac, replace=replace)
            metadata_.append(manip_metadata)
        metadata = pd.concat(metadata_).reset_index(drop=True)
        train_ds = VideoFact2Dataset(ds_root_dir, metadata, "train", crfs=["crf0", "crf23"], return_type="frame", return_frame_size=(512, 512), frame_size_op="resize")
        train_loader = DataLoader(
            train_ds,
            batch_size=self.training_config["batch_size"],
            shuffle=True,
            num_workers=self.training_config["num_workers"],
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        ds_root_dir = "/media/nas2/videofact2_data_large"
        ds_metadata_path = f"{ds_root_dir}/metadata.csv"
        metadata = pd.read_csv(ds_metadata_path)
        metadata = metadata.query("split == 'val'")
        manip_types = self.training_config["manip_types"]
        metadata_ = []
        for manip_type in manip_types:
            manip_metadata = metadata.query(f"vid_id.str.contains('{manip_type}')").sample(
                n=100, replace=True
            )
            metadata_.append(manip_metadata)
        metadata = pd.concat(metadata_).reset_index(drop=True)
        val_ds = VideoFact2Dataset(ds_root_dir, metadata, "val", crfs=["crf0", "crf23"], return_type="frame", return_frame_size=(512, 512), frame_size_op="resize")
        val_loader = DataLoader(
            val_ds,
            batch_size=self.training_config["batch_size"],
            shuffle=False,
            num_workers=self.training_config["num_workers"],
            persistent_workers=True,
            collate_fn=collate_fn,
        )
        return val_loader
