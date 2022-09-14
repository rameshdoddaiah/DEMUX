import torch
import torch.nn as nn
import numpy as np
from munch import Munch
from nte.utils import accuracy_softmax, accuracy_softmax_mse, confidence_score, Tee
# from nte.data.burst100 import BurstExistence, BurstLocation, BurstStrength, BurstFrequency, \
#     BurstTimeDifferenceExistence, BurstTimeDifferenceStrength
from torch.utils.data import SubsetRandomSampler
from nte.models import Linear, RNN
import json
import matplotlib.pyplot as plt
import ssl
import os
import wandb
import io
from PIL import Image


from nte.data.real.univariate.binary.cricketx.CricketXDataset import CricketXDataset
from nte.data.synth.blipmcshapelet.blipmcshapelet_dataset import BlipMCShapeletDataset
import seaborn as sns
sns.set_style("darkgrid")

torch.cuda.is_available = lambda: False


ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ENABLE_WANDB = False
# ENABLE_WANDB = True
WANDB_DRY_RUN = True
# WANDB_DRY_RUN = False

BASE_SAVE_DIR = 'results/0109/'
PROJECT_NAME = "blackbox_models"
# PROJECT_NAME = "time_series-cm-mse-lime-saliency"
TAG = 'TS_BBM'

if WANDB_DRY_RUN:
    os.environ["WANDB_MODE"] = "dryrun"

def train_black_box_model(dataset, hyper_params):

    if ENABLE_WANDB:
        wandb.init(entity="xai", project=PROJECT_NAME, name=hyper_params.model_name, tags=TAG,config=hyper_params,  reinit=True, force=True, dir=f"./wandb/{TAG}/")

    with Tee(filename=f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.log'):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        kwargs = {'num_workers': 1,
                  'pin_memory': True} if device == 'cuda' else {}
        # train_sampler = SubsetRandomSampler(dataset.train_data)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=10,
                                                #    kwargs=kwargs,
                                                   # sampler=train_sampler,
                                                   shuffle=False)

        # Create Model
        if hyper_params['model'] == 'rnn':
            model = RNN(hyper_params["rnn_config"]["ninp"], hyper_params["rnn_config"]["nhid"], "GRU",
                        hyper_params["rnn_config"]["nlayers"], hyper_params["rnn_config"]["nclasses"]).to(device)
        elif hyper_params['model'] == 'dnn':
            model = Linear(config=hyper_params).to(device)
        else:
            raise Exception(f"Unknown model {use_model}")

        print("Config: \n", json.dumps(hyper_params, indent=2))
        print("Model: \n", model)

        # Loss and optimizer
        if hyper_params['loss'] == 'ce':
            criterion = nn.NLLLoss()
            activation = torch.log_softmax
        elif hyper_params['loss'] == 'mse':
            criterion = nn.MSELoss()
            activation = torch.softmax
        else:
            raise Exception('Unknown loss {loss}')
        # Train the model
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyper_params.learning_rate)

        step_counter = 0

        cost, acc, c_scores, preds_0, preds_1 = [], [], [], [], []

        for epoch in range(hyper_params.num_epochs):
            batch_cost, batch_acc, batch_c_scores, batch_preds_0, batch_preds_1 = [], [], [], [], []
            final_loss = 0.0
            for i, (X, y) in enumerate(train_loader):

                if hyper_params['model'] == 'rnn':
                    X = torch.tensor(X.reshape(-1, hyper_params["timesteps"],
                                               hyper_params["rnn_config"]["ninp"]), dtype=torch.float32)
                elif hyper_params['model'] == 'dnn':
                    X = torch.tensor(X, dtype=torch.float32)

                if hyper_params['loss'] == 'ce':
                    y = torch.tensor(y, dtype=torch.long).reshape([-1])
                elif hyper_params['loss'] == 'mse':
                    y = np.eye(hyper_params['num_classes'])[y]
                    y = torch.tensor(y, dtype=torch.float).reshape([-1, 2])

                X = X.to(device)
                y = y.to(device)
                # Forward pass
                y_hat = model(X)
                optimizer.zero_grad()

                loss = criterion(activation(y_hat, dim=1), y)                
                loss.backward()
                optimizer.step()
                batch_cost.append(loss.item())
                scores, cs = confidence_score(torch.softmax(y_hat, dim=1).cpu().detach().numpy(), y.cpu().detach().numpy())
                batch_c_scores.append(cs)

                if 0 in scores:
                    batch_preds_0.append(scores[0])
                    # wandb.log({"Prediction Scores 0": scores[0]})
                if 1 in scores:
                    batch_preds_1.append(scores[1])
                    # wandb.log({"Prediction Scores 1": scores[1]})
                batch_acc.append(accuracy_softmax(y, y_hat))

                # wandb.log({"Loss":loss})
                # wandb.log({"Confidence Score":cs})
                
            cost.append(np.mean(batch_cost))
            acc.append(np.mean(batch_acc))
            c_scores.append(np.mean(batch_c_scores))
            preds_0.append(np.mean(np.array(batch_preds_0)))
            preds_1.append(np.mean(np.array(batch_preds_1)))

            if ENABLE_WANDB:
                wandb.log({"BBM Epoch": epoch})
                wandb.log({"BMM Loss ": float(cost[-1])})
                wandb.log({"BBM Accuracy": float(acc[-1])})
                wandb.log({"BBM Confidence_scores": float(c_scores[-1])})
                wandb.log({"BBM Class 0 Pred ": float(preds_0[-1])})
                wandb.log({"BBM Class 1 Pred ": float(preds_1[-1])})

            # if (i + 1) % 1 == 0:
            print('Epoch [{}/{}] | Loss: {:.4f} |  Accuracy: {:.4f} | Prediction Confidence: {:.2f} | Class 0:{:.4f} | Class 1:{:.4f}'
                  .format(epoch + 1, hyper_params.num_epochs, cost[-1], acc[-1], c_scores[-1], preds_0[-1], preds_1[-1]))

        fig, ax = plt.subplots(1, 2, figsize=(20,6))        
        ax[0].plot(cost, label="BBM cost")
        ax[0].plot(acc, label="BBM accuracy")
        ax[0].plot(c_scores, label="BBM confidence")
        ax[0].legend()
        ax[1].plot(preds_0, label="Prediction class 0 confidence")
        ax[1].plot(preds_1, label="Prediction class 1 confidence")
        ax[1].legend()
        plt.show()

        if ENABLE_WANDB:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            wp = wandb.Image(Image.open(buf), caption="BBM plots")

            # wandb.log({'Cost Accuracy Confidence Preds': plt})
            wandb.log({"Result":[wp]})
            # wandb.Image(plt)

        # Test the model
        if hyper_params['model'] == 'rnn':
            y_hat = model(
                torch.tensor(dataset.test_data.reshape(-1, hyper_params["timesteps"],
                                                       hyper_params["rnn_config"]["ninp"]), dtype=torch.float32).to(device))
        elif hyper_params['model'] == 'dnn':
            y_hat = model(torch.tensor(dataset.test_data, dtype=torch.float32).to(device))

        if hyper_params['loss'] == 'ce':
            labels = torch.tensor(dataset.test_label,
                                  dtype=torch.long).reshape([-1]).to(device)
            test_acc = accuracy_softmax(labels, y_hat)
        elif hyper_params['loss'] == 'mse':
            labels = torch.tensor(np.eye(np.max(
                dataset.test_label) - np.min(dataset.test_label) + 1)[dataset.test_label]).to(device)
            labels = torch.tensor(labels, dtype=torch.float).reshape([-1, 2]).to(device)
            test_acc = accuracy_softmax_mse(labels, y_hat)

        print('Test Accuracy {} | Confidence {}'.format(100 * test_acc,
                                                        confidence_score(torch.softmax(y_hat, dim=1).cpu().detach().numpy(), dataset.test_label)))

        # Save the model checkpoint
        torch.save(
            model, f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.ckpt')
        return model


if __name__ == '__main__':

    # for debugging...
    # import debugpy
    # os. chdir("/work/rdoddaiah/TimeSeriesSaliencyMaps/nte")

    # use_model = 'rnn'
    use_model = 'dnn'
    use_loss = 'ce'

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = BlipV3Dataset()
    # dataset = BurstExistence10()
    # dataset = BurstLocation10()
    # dataset = BurstFrequency10()
    # dataset = BurstStrength10()
    # dataset = BurstTimeDifferenceExistence10()
    # dataset = BurstTimeDifferenceStrength10()
    dataset = BlipMCShapeletDataset()

    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    # dataset = CricketXDataset()
    TIMESTEPS = len(dataset.train_data[0])
    model_name = f"{dataset.name}_{use_model}_{use_loss}"

    # Hyper-parameters
    hyper_params = Munch({
        "model_save_path": "./",
        "model": use_model,
        "loss": use_loss,
        "model_name": model_name,
        "dependency_meta": "",
        "timesteps": TIMESTEPS,
        "num_classes": 2,
        "rnn_config": {
            "ninp": 1,
            "nhid": 10,
            "nlayers": 1,
            "nclasses": 2
        },
        "dnn_config": {
            "layers": [10, 10, 2],
        },
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-3
    })

    train_black_box_model(dataset=dataset, hyper_params=hyper_params)
