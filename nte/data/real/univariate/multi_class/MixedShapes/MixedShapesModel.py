from nte.models import Model
from nte.data import MixedShapes_MultiClass
import torch.nn as nn
import torch
from functools import reduce
import operator
import warnings
from nte import fit, NTE_TRAINED_MODEL_PATH
warnings.filterwarnings("ignore")
torch.cuda.is_available = lambda: False

class MixedShapesCNNModel(Model):
    def __init__(self, config):
        super().__init__(config=config)
        self.timesteps = config["timesteps"]

        self.sigmoid_activation = nn.Sigmoid()
        self.softmax_activation = nn.Softmax(dim=-1)
        self.tanh_activation = torch.nn.Tanh()
        self.relu_activation = torch.nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=1,
                               kernel_size=(1, 1))
        nodes = reduce(operator.mul, self.conv1(torch.rand([1, 1, 1, self.timesteps])).shape[1:])
        self.linear1 = nn.Linear(nodes, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.x = self.x.reshape([-1, 1, 1, self.timesteps])
        self.x = self.tanh_activation(self.conv1(self.x))
        self.x = self.x.reshape([-1, reduce(operator.mul, self.x.shape[1:])])
        self.x = self.tanh_activation(self.linear1(self.x))
        self.x = self.linear2(self.x)
        return self.x

    def evaluate(self, data, args):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = torch.softmax(self.forward(data.reshape([-1, self.timesteps])), dim=-1)
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.cpu().detach().numpy(), predicted_class.cpu().detach().numpy(), prediction_probabilities.cpu().detach().numpy()


if __name__ == '__main__':
    dataset = MixedShapes_MultiClass()
    config = {
        "model_name": "CNN",
        "timesteps": len(dataset.train_data[0]),
        "dependency_meta": "",
    }
    model = MixedShapesCNNModel(config)
    hyper_params = {
        "loss": "mse",
        "learning_rate": 5e-3,
        "num_epochs": 150,
        "num_classes": dataset.num_classes,
        "batch_size": 10,
        "model_save_path": NTE_TRAINED_MODEL_PATH+"/all_bbms/",
        "model_name": f"{dataset.name}_cnn_mse"
    }
    fit(dataset, model, hyper_params)
