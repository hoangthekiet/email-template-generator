import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class ColorModel(nn.Module):
    """
    Simple Multi-layer Perceptron with 2 fully connected layers.
    Batch-norm is applied to correlate the data points within each batch.

    Input
    -----
    - Dim = 4
    - Features: hue, saturation, lightness, frequency

    Output
    ------
    - Dim = 5
    - Features: backdrop, canvas, font, link, accent
    """
    def __init__(self, hidden_size: int = 8):
        super(ColorModel, self).__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 5)
        self.bn1 = nn.BatchNorm1d(5)
        self.smax = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(self.fc2(x))
        return self.smax(x)


def train(dataloader: DataLoader, lr: float = 0.003, num_epochs: int = 1000) -> ColorModel:
    from torch.utils.tensorboard import SummaryWriter
    
    # Init model, define loss function and optimizer
    model = ColorModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()

    # Training loop
    for epoch in range(num_epochs):
        sum_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            loss.backward()
            optimizer.step()

        writer.add_scalars("Loss", sum_loss, epoch)
        print(f'Epoch #[{epoch+1}/{num_epochs}], Loss: {sum_loss:.4f}')    
    
    writer.flush()
    writer.close()
    
    return model