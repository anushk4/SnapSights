import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()

        # Define the layers within the model
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding="same")
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.LeakyReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.LeakyReLU()
        self.pool5 = nn.MaxPool2d(2, 2)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding="same")
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU()
        self.pool6 = nn.MaxPool2d(2, 2)
        
        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, padding="same")
        self.bn7 = nn.BatchNorm2d(1024)
        self.relu7 = nn.LeakyReLU()
        self.pool7 = nn.MaxPool2d(2, 2)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(1024, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.relu_fc1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))
        
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu_fc1(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
