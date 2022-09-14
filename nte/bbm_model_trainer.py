from nte.models import Model
import numpy as np
import torch.nn as nn
import torch
from nte.utils import accuracy_softmax, accuracy_softmax_mse, confidence_score, Tee
from rainbow_print import printr
from functools import reduce
import operator
import warnings
from collections import defaultdict
import json
from nte.models import CNN
from nte.data.real.univariate.multi_class.ACSF1.ACSFDataset_MultiClass import ACSFDataset_MultiClass
from nte.data.real.univariate.multi_class.Meat.MeatDataset_MultiClass import MeatDataset_MultiClass
from nte import NTE_TRAINED_MODEL_PATH

warnings.filterwarnings("ignore")
torch.cuda.is_available = lambda: False


def fit(dataset, model, hyper_params, run_test=True, save_model=True):
    print("Hyperparams: \n")
    print(json.dumps(hyper_params, indent=2))
    print("\n\n")

    # Loss and optimizer
    if hyper_params['loss'] == 'ce':
        criterion = nn.NLLLoss()
        activation = torch.log_softmax
        accuracy = accuracy_softmax
    elif hyper_params['loss'] == 'mse':
        criterion = nn.MSELoss()
        activation = torch.softmax
        accuracy = accuracy_softmax_mse
    else:
        raise Exception('Unknown loss {loss}')

    # Train the model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyper_params["learning_rate"],  weight_decay=1e-5)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=hyper_params["batch_size"],
                                               #kwargs=kwargs,
                                               # sampler=train_sampler,
                                               shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = defaultdict(lambda: [])

    for epoch in range(hyper_params["num_epochs"]):
        metrics["batch_cost"], metrics["batch_acc"], metrics["batch_conf"] = [], [], []

        for i, (X, y) in enumerate(train_loader):
            oy = y
            if hyper_params['loss'] == 'ce':
                y = torch.tensor(y, dtype=torch.long).reshape([-1])
            elif hyper_params['loss'] == 'mse':
                y = np.eye(hyper_params['num_classes'])[y]
                y = torch.tensor(y, dtype=torch.float).reshape([-1, hyper_params['num_classes']])

            X, y = X.to(device), y.to(device)

            # Forward pass
            y_hat = model(X)

            # Backward
            optimizer.zero_grad()
            loss = criterion(activation(y_hat, dim=1), y)
            scores, cs = confidence_score(torch.softmax(y_hat, dim=1).cpu().detach().numpy(), oy)
            loss.backward()
            optimizer.step()

            # Metrics
            metrics["batch_cost"].append(loss.item())
            metrics["batch_conf"].append(cs)
            metrics["batch_acc"].append(accuracy(y, y_hat))

            # wandb.log({"Loss":loss})
            # wandb.log({"Confidence Score":cs})

        metrics["cost"].append(np.mean(metrics["batch_cost"]))
        metrics["acc"].append(np.mean(metrics["batch_acc"]))
        metrics["conf"].append(np.mean(metrics["batch_conf"]))

        # if ENABLE_WANDB:
        #     wandb.log({"BBM Epoch": epoch})
        #     wandb.log({"BMM Loss ": float(cost[-1])})
        #     wandb.log({"BBM Accuracy": float(acc[-1])})
        #     wandb.log({"BBM Confidence_scores": float(c_scores[-1])})
        #     wandb.log({"BBM Class 0 Pred ": float(preds_0[-1])})
        #     wandb.log({"BBM Class 1 Pred ": float(preds_1[-1])})

        # if (i + 1) % 1 == 0:
        printr(f'Epoch [{epoch + 1}/{hyper_params["num_epochs"]}] {(epoch + 1)/hyper_params["num_epochs"]*100:.2f}% | Loss: {metrics["cost"][-1]:.4f} |  Accuracy: { metrics["acc"][-1]:.2f} | Confidence: {metrics["conf"][-1]:.2f}', sep="|")

    # Test the model
    if run_test:
        y_hat = model(torch.tensor(dataset.test_data, dtype=torch.float32).to(device))
        y_hat = activation(y_hat, dim=1)

        if hyper_params['loss'] == 'ce':
            labels = torch.tensor(dataset.test_label,
                                dtype=torch.long).reshape([-1, 1]).to(device)
            test_acc = accuracy_softmax(labels, y_hat)
        elif hyper_params['loss'] == 'mse':
            labels = torch.tensor(np.eye(np.max(
                dataset.test_label) - np.min(dataset.test_label) + 1)[dataset.test_label]).to(device)
            labels = torch.tensor(labels, dtype=torch.float).reshape([-1, hyper_params["num_classes"]]).to(device)
            test_acc = accuracy_softmax_mse(labels, y_hat)

        print('Test Accuracy {} | Confidence {}'.format(100 * test_acc,
                                                        {k: round(v.item(), 2) for k, v in confidence_score(torch.softmax(y_hat, dim=1).cpu().detach().numpy(), dataset.test_label)[0].items()}))

    # Save the model checkpoint
    if save_model:
        # with torch.no_grad():
        #     traced_cell = torch.jit.trace(model, (X))
        # torch.jit.save(traced_cell, f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.ckpt')
        torch.save(model, f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.ckpt')
        # torch.save(dict(
        #     model=model,
        #     model_state=model.state_dict()), f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.ckpt')


if __name__ == "__main__":
    dataset = MeatDataset_MultiClass()
    config = {
        "model_name": "ce",
        "timesteps": len(dataset.train_data[0]),
        "num_classes": dataset.num_classes,
        "dependency_meta": "",
    }
    model = CNN(config)
    hyper_params = {
        "loss": "ce",
        "learning_rate": 1e-3,
        "num_epochs": 500,
        "num_classes": dataset.num_classes,
        "batch_size": 32,
        "model_save_path": NTE_TRAINED_MODEL_PATH+"/all_bbms/",
        "model_name": f"{dataset.name}_cnn_mse"
    }
    fit(dataset, model, hyper_params)
