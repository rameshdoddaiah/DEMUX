import torch
import torch.nn as nn
import numpy as np
import os
from munch import Munch
from nte.utils import accuracy_softmax, accuracy_softmax_mse, confidence_score, Tee
from nte.data.synth.burst10 import BurstLocation10
# from nte.data.burst100 import BurstExistence, BurstLocation, BurstStrength, BurstFrequency, \
#     BurstTimeDifferenceExistence, BurstTimeDifferenceStrength
from torch.utils.data import SubsetRandomSampler
from nte.models import Linear, RNN
import json

def train_black_box_model(dataset, hyper_params):
    with Tee(filename=f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.log'):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # train_sampler = SubsetRandomSampler(dataset.train_data)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=10,
                                                   # sampler=train_sampler,
                                                   shuffle=False)

        # Create Model
        if hyper_params['model'] == 'rnn':
            model = RNN(hyper_params["rnn_config"]["ninp"], hyper_params["rnn_config"]["nhid"], "GRU",
                    hyper_params["rnn_config"]["nlayers"], hyper_params["rnn_config"]["nclasses"])
        elif hyper_params['model'] == 'dnn':
            model = Linear(config=hyper_params)
        else:
            raise Exception(f"Unknow model {use_model}")


        print("Config: \n", json.dumps(hyper_params, indent=2))
        print("Model: \n", model)


        # Loss and optimizer
        if use_loss == 'ce':
            criterion = nn.NLLLoss()
            activation = torch.log_softmax
        elif use_loss == 'mse':
            criterion = nn.MSELoss()
            activation = torch.softmax
        else:
            raise Exception('Unknown loss {loss}')
        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.learning_rate)

        step_counter = 0
        for epoch in range(hyper_params.num_epochs):
            batch_cost, batch_acc, c_scores, preds_0, preds_1 = [], [], [], [], []
            final_loss = 0.0
            for i, (X, y) in enumerate(train_loader):


                if use_model == 'rnn':
                    X = torch.tensor(X.reshape(-1, hyper_params["timesteps"],
                                               hyper_params["rnn_config"]["ninp"]), dtype=torch.float32)
                elif use_model == 'dnn':
                    X = torch.tensor(X, dtype=torch.float32)

                if use_loss == 'ce':
                    y = torch.tensor(y, dtype=torch.long).reshape([-1])
                elif use_loss == 'mse':
                    y = np.eye(hyper_params['num_classes'])[y]
                    y = torch.tensor(y, dtype=torch.float).reshape([-1, 2])

                # Forward pass
                y_hat = model(X)
                optimizer.zero_grad()

                loss = criterion(activation(y_hat, dim=1), y)
                final_loss += loss.item()
                loss.backward()
                optimizer.step()
                batch_cost.append(loss.item())
                scores, cs = confidence_score(torch.softmax(y_hat, dim=1).detach().numpy(), y.detach().numpy())
                c_scores.append(cs)

                if 0 in scores:
                    preds_0.append(scores[0])
                if 1 in scores:
                    preds_1.append(scores[1])
                batch_acc.append(accuracy_softmax(y, y_hat))

            # if (i + 1) % 1 == 0:
            print('Epoch [{}/{}] | Loss: {:.4f} |  Accuracy: {:.4f} | Prediction Confidence: {:.2f} | Class 0:{:.4f} | Class 1:{:.4f}'
                  .format(epoch + 1, hyper_params.num_epochs, np.mean(batch_cost), np.mean(batch_acc), np.mean(c_scores),
                          np.mean(np.array(preds_0)), np.mean(np.array(preds_1))))

        # Test the model

        # Test the model
        if use_model == 'rnn':
            y_hat = model(
                torch.tensor(dataset.test_data.reshape(-1, hyper_params["timesteps"],
                                       hyper_params["rnn_config"]["ninp"]), dtype=torch.float32))
        elif use_model == 'dnn':
            y_hat = model(torch.tensor(dataset.test_data, dtype=torch.float32))

        if use_loss == 'ce':
            labels = torch.tensor(dataset.test_label, dtype=torch.long).reshape([-1])
            test_acc = accuracy_softmax(labels, y_hat)
        elif use_loss == 'mse':
            labels = torch.tensor(np.eye(np.max(dataset.test_label) - np.min(dataset.test_label) + 1)[dataset.test_label])
            labels = torch.tensor(labels, dtype=torch.float).reshape([-1, 2])
            test_acc = accuracy_softmax_mse(labels, y_hat)

        print('Test Accuracy {} | Confidence {}'.format( 100 * test_acc, confidence_score(torch.softmax(y_hat, dim=1).detach().numpy(), dataset.test_label)))

        # Save the model checkpoint
        torch.save(model, f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.ckpt')


if __name__ == '__main__':

    # # # debugpy.listen(('10.185.0.10', 5678))
    # # # debugpy.wait_for_client()

    # # debugpy.listen(5699)
    # debugpy.connect(5698)
    # print("Waiting for debugger attach")
    # # debugpy.wait_for_client()
    # debugpy.breakpoint()
    # print('break on this line')

    # Configuration
    # filename = __file__
    # print("starting debugpy")
    # pdb.set_trace()
    # print(f'path = {filename}')
    use_model = 'rnn'
    # use_model = 'dnn'
    use_loss = 'ce'

    os. chdir("/work/rdoddaiah/TimeSeriesSaliencyMaps/nte")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = BurstExistence10()
    dataset = BurstLocation10()
    # dataset = BurstFrequency10()
    # dataset = BurstStrength10()
    # dataset = BurstTimeDifferenceExistence10()
    # dataset = BurstTimeDifferenceStrength10()

    model_name = f"{dataset.name}_{use_model}_{use_loss}"

    # Hyper-parameters
    hyper_params = Munch({
        "model_save_path": "./trained_models/burst10/",
        "model": use_model,
        "loss": use_loss,
        "model_name": model_name,
        "dependency_meta": dataset.train_meta,
        "timesteps": 10,
        "num_classes":2,
        "rnn_config": {
            "ninp": 1,
            "nhid": 10,
            "nlayers": 1,
            "nclasses": 2
        },
        "dnn_config":{
            "layers":[50, 20, 2],
        },
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-4
    })

    train_black_box_model(dataset=dataset, hyper_params=hyper_params)
