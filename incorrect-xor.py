import torch
from torch import nn, optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

print("PyTorch version:", torch.__version__)
print("Torch Lightning version:", pl.__version__)

# Check if MPS is available and set the device accordingly
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

# XOR inputs and targets
xor_inputs = [
    Variable(torch.Tensor([0, 0]).to(device)),
    Variable(torch.Tensor([0, 1]).to(device)),
    Variable(torch.Tensor([1, 0]).to(device)),
    Variable(torch.Tensor([1, 1]).to(device)),
]
xor_targets = [
    Variable(torch.Tensor([0]).to(device)),
    Variable(torch.Tensor([1]).to(device)),
    Variable(torch.Tensor([1]).to(device)),
    Variable(torch.Tensor([0]).to(device)),
]
xor_data = list(zip(xor_inputs, xor_targets))
train_loader = DataLoader(xor_data, batch_size=1, shuffle=True)

class XORModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(2, 4)
        self.output_layer = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, input):
        x = self.input_layer(input)
        x = self.sigmoid(x)
        x = self.output_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        outputs = self(xor_input)
        loss = self.loss(outputs, xor_target)
        return loss

checkpoint_callback = ModelCheckpoint()
model = XORModel().to(device)
trainer = pl.Trainer(max_epochs=300, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader)

trained_model = XORModel.load_from_checkpoint(checkpoint_callback.best_model_path).to(device)

for val in xor_inputs:
    print("Input:", val.to(device))
    output = trained_model(val)
    print("Output:", output)
