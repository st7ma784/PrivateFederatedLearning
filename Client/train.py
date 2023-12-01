import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import random
class PermutationLayer(nn.Module):
    def __init__(self,key):

        super(PermutationLayer, self).__init__()
        # Define your permutation layer logic here
        self.key = key
        #the key is a list of 0s and 1s that is used to determine the permutation,
        #so the permutation is the same for each key
        #for example, if the key is '101', then the permutation is [0,2,1]
        #if the key is '010', then the permutation is [1,0,2]
        #make out seed by converting the key to an integer as a binary number
        asbool= [bool(int(i)) for i in key]
        #convert the list of booleans to a binary number
        asint= int(''.join([str(int(i)) for i in asbool]),2)

        #we use a random generator to generate the permutation but seed it with the number of trues in the key
        self.generator = torch.Generator().manual_seed(asint)
        # let n be the number of trues in the key
        n= sum(key)
        perm= torch.randperm(n,generator=self.generator)
        self.permutations = torch.arange(len(self.key))
        self.permutations= torch.where(key, self.permutations[self.key][perm],self.permutations)

    def forward(self, x):
        # Implement the forward pass of your permutation layer here
        return x[self.permutations]

class CLIPModel(pl.LightningModule):
    def __init__(self):
        super(CLIPModel, self).__init__()
        # Define your CLIP model architecture here
        self.permutation_layer = PermutationLayer()
        
    def forward(self, x):
        # Implement the forward pass of your CLIP model here
        pass
    
    def training_step(self, batch, batch_idx):
        # Implement the training step logic here
        pass
    
    def validation_step(self, batch, batch_idx):
        # Implement the validation step logic here
        # Send the model weights to the server and receive deltas
        weights = self.state_dict()
        deltas = self.send_weights_to_server(weights)
        return deltas
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        return optimizer


if __name__ == '__main__':
    import sys
    #get key for model from first CLI argument
    key = sys.argv[1]
    from dataset import *    
    data_module = CocoDataModule(data_dir='./data', batch_size=32)
    model = CLIPModel(key)
    trainer = pl.Trainer()
    trainer.fit(model)
model = CLIPModel()

# Load the weights from the server
weights = load_weights_from_server()

# Set the model weights
model.load_state_dict(weights)

# Train the model
trainer = pl.Trainer()
trainer.fit(model)

# Apply the received deltas to update the model weights
deltas = model.validation_step(None, None)
model.load_state_dict(apply_deltas(model.state_dict(), deltas))
