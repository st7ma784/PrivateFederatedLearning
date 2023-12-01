
import pytorch_lightning as pl
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # Download or preprocess the COCO dataset if needed
        CocoDetection(root=self.data_dir, annFile='annotations.json', transform=None)

    def setup(self, stage=None):
        # Load the COCO dataset
        self.dataset = CocoDetection(root=self.data_dir, annFile='annotations.json', transform=None)

    def train_dataloader(self):
        # Create a DataLoader for training data
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # Create a DataLoader for validation data
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # Create a DataLoader for test data
        return DataLoader(self.dataset, batch_size=self.batch_size)
