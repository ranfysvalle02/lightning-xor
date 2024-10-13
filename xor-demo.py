import torch
from torch import nn, optim
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

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
        # Input layer: 2 input features, 8 neurons in hidden layer
        self.input_layer = nn.Linear(2, 8)
        # Hidden layer: 8 neurons in, 4 neurons out
        self.hidden_layer = nn.Linear(8, 4)
        # Output layer: 4 neurons in, 1 output
        self.output_layer = nn.Linear(4, 1)
        # ReLU for hidden layers
        self.relu = nn.ReLU()
        # Loss function: Binary Cross Entropy with Logits
        self.loss = nn.BCEWithLogitsLoss()

        # Initialize weights explicitly
        torch.nn.init.xavier_uniform_(self.input_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, input):
        x = self.input_layer(input)
        x = self.relu(x)  # Use ReLU for better gradient flow
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x  # No sigmoid here because BCEWithLogitsLoss handles it

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)  # Increased LR
        return optimizer

    def training_step(self, batch, batch_idx):
        xor_input, xor_target = batch
        outputs = self(xor_input)
        loss = self.loss(outputs, xor_target)
        self.log('train_loss', loss)
        return loss

# Initialize the model and move to the same device
model = XORModel().to(device)

# Train the model
checkpoint_callback = ModelCheckpoint()
trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback])  # Increased epochs
trainer.fit(model, train_loader)

print("Training Complete")
print("Best model path:", checkpoint_callback.best_model_path)

# Load the best model checkpoint
trained_model = XORModel.load_from_checkpoint(checkpoint_callback.best_model_path).to(device)

# Test the model with inputs
for val in xor_inputs:
    val = val.to(device)
    output = torch.sigmoid(trained_model(val))  # Apply sigmoid at test time
    print("Input:", val)
    print("Output:", output)
