
import pytorch_lightning as pl
from torchvision.datasets import CocoCaptions
from torch.utils.data import DataLoader
import torch
from clip.simple_tokenizer import SimpleTokenizer

class TokenizedCocoCaptions(CocoCaptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = SimpleTokenizer()
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        tokenized_target = self.tokenizer.tokenize(target)
        return img, tokenized_target


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        #use tokenizers to convert the captions into a format that can be used by the CLIP model

    def prepare_data(self):
        #check that the data is available
        TokenizedCocoCaptions(root=self.data_dir, annFile='annotations.json', transform=None)


    def setup(self, stage=None):
        # Load the COCO dataset
        # to properly test this federated learning example, we would need to split the dataset into a subset for each client
        # for simplicity, we will just use a random 50% of the dataset for each client
        dataset=TokenizedCocoCaptions(root=self.data_dir, annFile='annotations.json', transform=None)

        self.dataset,_ =  torch.utils.data.random_split(dataset, [len(dataset)//2, len(dataset)-len(dataset)//2])
        

    def train_dataloader(self):
        # Create a DataLoader for training data
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # Create a DataLoader for validation data
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # Create a DataLoader for test data
        return DataLoader(self.dataset, batch_size=self.batch_size)
