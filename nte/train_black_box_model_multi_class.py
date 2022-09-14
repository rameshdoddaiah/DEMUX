import torch
import torch.nn as nn
import numpy as np
from munch import Munch
from nte.utils import accuracy_softmax, accuracy_softmax_mse, confidence_score, Tee
# from nte.data.synth.blipv3 import BlipV3Dataset
# from nte.data.synth.burst10.location import BurstLocation10
# from nte.data.burst100 import BurstExistence, BurstLocation, BurstStrength, BurstFrequency, \
#     BurstTimeDifferenceExistence, BurstTimeDifferenceStrength
from torch.utils.data import SubsetRandomSampler
from nte.models import Linear, RNN
import json
import ssl
import os
import wandb
import io
from PIL import Image
from nte.data.real.univariate.multi_class.cricketx.CricketXDataset_MultiClass import CricketXDataset_MultiClass
from nte.data.real.univariate.multi_class.AbnormalHeartbeat.AbnormalHeartbeatDataset_MultiClass  import AbnormalHeartbeatDataset_MultiClass
from nte.data.real.univariate.multi_class.ACSF1.ACSFDataset_MultiClass import ACSFDataset_MultiClass
from nte.data.real.univariate.multi_class.EOGHorizontalSignal.EOGHorizontalSignalDataset_MultiClass import EOGHorizontalSignalDataset_MultiClass
from nte.data.real.univariate.multi_class.Plane.PlaneDataset_MultiClass import PlaneDataset_MultiClass
from nte.data.real.univariate.multi_class.Trace.TraceDataset_MultiClass import TraceDataset_MultiClass
from nte.data.real.univariate.multi_class.Meat.MeatDataset_MultiClass import MeatDataset_MultiClass
from nte.data.real.univariate.multi_class.ECG5000.ECG5000Dataset_MultiClass import ECG5000Dataset_MultiClass
#from nte.data.real.univariate.multi_class.Rock.RockDataset_MultiClass import RockDataset_MultiClass
from nte.data.real.univariate.multi_class.SmallKitchenAppliances.SmallKitchenAppliancesDataset_MultiClass import SmallKitchenAppliancesDataset_MultiClass
from nte.data.synth.blipmc.blipmc_dataset import BlipMCDataset
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt

#torch.cuda.is_available = lambda: False

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

def train_black_box_model_multi_class(dataset, hyper_params):

    if ENABLE_WANDB:
        wandb.init(entity="xai", project=PROJECT_NAME, name=hyper_params.model_name, tags=TAG,config=hyper_params,  reinit=True, force=True, dir=f"./wandb/{TAG}/")

    with Tee(filename=f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.log'):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
        # train_sampler = SubsetRandomSampler(dataset.train_data)
        train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=10,
                                                   # kwargs=kwargs,
                                                   # sampler=train_sampler,
                                                   shuffle=False)
                                                #    shuffle=True)

        # Create Model
        if hyper_params['model'] == 'rnn':
            model = RNN(hyper_params["rnn_config"]["ninp"], hyper_params["rnn_config"]["nhid"], "GRU",hyper_params["rnn_config"]["nlayers"], hyper_params["rnn_config"]["nclasses"]).to(device)
        elif hyper_params['model'] == 'dnn':
            model = Linear(config=hyper_params).to(device)
        else:
            raise Exception(f"Unknown model {use_model}")

        print("Config: \n", json.dumps(hyper_params, indent=2))
        print("Model: \n", model)

        # Loss and optimizer
        if hyper_params['loss'] == 'ce':
            criterion = nn.NLLLoss()
            # activation = torch.log_softmax
            activation = nn.LogSoftmax(dim=1)
            # activation = torch.log
        elif hyper_params['loss'] == 'mse':
            criterion = nn.MSELoss()
            activation = torch.softmax
        else:
            raise Exception('Unknown loss {loss}')
        # Train the model
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyper_params.learning_rate)

        step_counter = 0

        cost, acc, c_scores, preds_0, preds_1, preds_2,preds_3,preds_4,preds_5,preds_6,preds_7,preds_8,preds_9, preds_10, preds_11 = [], [], [], [], [],[],[],[],[],[],[],[],[],[],[]

        for epoch in range(hyper_params.num_epochs):
            batch_cost, batch_acc, batch_c_scores, batch_preds_0, batch_preds_1, batch_preds_2,batch_preds_3,batch_preds_4,batch_preds_5,batch_preds_6,batch_preds_7,batch_preds_8,batch_preds_9,batch_preds_10, batch_preds_11\
                = [], [], [], [], [],[],[],[],[],[],[],[],[],[],[]
            final_loss = 0.0
            for i, (X, y) in enumerate(train_loader):

                if hyper_params['model'] == 'rnn':
                    X = torch.tensor(X.reshape(-1, hyper_params["timesteps"],hyper_params["rnn_config"]["ninp"]), dtype=torch.float32)
                elif hyper_params['model'] == 'dnn':
                    X = torch.tensor(X, dtype=torch.float32)

                if hyper_params['loss'] == 'ce':
                    y = torch.tensor(y, dtype=torch.long).reshape([-1])
                elif hyper_params['loss'] == 'mse':
                    y = np.eye(hyper_params['num_classes'])[y]
                    y = torch.tensor(y, dtype=torch.float).reshape([-1, 2])


                X = X.to(device)
                y = y.to(device)
                # y_one_hot = torch.nn.functional.one_hot(y)
                # Forward pass
                y_hat = model(X)
                optimizer.zero_grad()

                loss = criterion(activation(y_hat), y)
                # loss = criterion(activation(y_hat, dim=1), y)
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
                if 2 in scores:
                    batch_preds_2.append(scores[2])
                if 3 in scores:
                    batch_preds_3.append(scores[3])
                if 4 in scores:
                    batch_preds_4.append(scores[4])
                if 5 in scores:
                    batch_preds_5.append(scores[5])
                if 6 in scores:
                    batch_preds_6.append(scores[6])
                if 7 in scores:
                    batch_preds_7.append(scores[7])
                if 8 in scores:
                    batch_preds_8.append(scores[8])
                if 9 in scores:
                    batch_preds_9.append(scores[9])
                if 10 in scores:
                    batch_preds_10.append(scores[10])
                if 11 in scores:
                    batch_preds_11.append(scores[11])
                batch_acc.append(accuracy_softmax(y, y_hat))

                # wandb.log({"Loss":loss})
                # wandb.log({"Confidence Score":cs})
            cost.append(np.mean(batch_cost))
            acc.append(np.mean(batch_acc))
            c_scores.append(np.mean(batch_c_scores))
            preds_0.append(np.mean(np.array(batch_preds_0)))
            preds_1.append(np.mean(np.array(batch_preds_1)))
            preds_2.append(np.mean(np.array(batch_preds_2)))
            preds_3.append(np.mean(np.array(batch_preds_3)))
            preds_4.append(np.mean(np.array(batch_preds_4)))
            preds_5.append(np.mean(np.array(batch_preds_5)))
            preds_6.append(np.mean(np.array(batch_preds_6)))
            preds_7.append(np.mean(np.array(batch_preds_7)))
            preds_8.append(np.mean(np.array(batch_preds_8)))
            preds_9.append(np.mean(np.array(batch_preds_9)))
            preds_10.append(np.mean(np.array(batch_preds_10)))
            preds_11.append(np.mean(np.array(batch_preds_11)))

            if ENABLE_WANDB:
                wandb.log({"BBM Epoch": epoch})
                wandb.log({"BMM Loss ": float(cost[-1])})
                wandb.log({"BBM Accuracy": float(acc[-1])})
                wandb.log({"BBM Confidence_scores": float(c_scores[-1])})
                wandb.log({"BBM Class 0 Pred ": float(preds_0[-1])})
                wandb.log({"BBM Class 1 Pred ": float(preds_1[-1])})

            # if (i + 1) % 1 == 0:
            print('Epoch[{}/{}] | Loss:{:.4f} |Acc: {:.4f}|Pred Confidence: {:.2f} | C 0:{:.4f} | C 1:{:.4f} | C 2:{:.4f}| C 3:{:.4f}| C 4:{:.4f}| C 5:{:.4f}| C 6:{:.4f}| C 7:{:.4f}| C 8:{:.4f}| C 9:{:.4f}| C 10:{:.4f}| C 11:{:.4f}'
                  .format(epoch + 1, hyper_params.num_epochs, cost[-1], 100*acc[-1], c_scores[-1], preds_0[-1], \
                      preds_1[-1],preds_2[-1],preds_3[-1],preds_4[-1],preds_5[-1],preds_6[-1],preds_7[-1],preds_8[-1],preds_9[-1],preds_10[-1],preds_11[-1]))

        fig, ax = plt.subplots(1, 2, figsize=(20,6))
        ax[0].plot(cost, label="BBM cost")
        ax[0].plot(acc, label="BBM accuracy")
        ax[0].plot(c_scores, label="BBM confidence")
        ax[0].legend()
        ax[1].plot(preds_0, label="Prediction class 0 confidence")
        ax[1].plot(preds_1, label="Prediction class 1 confidence")
        ax[1].legend()
        plt.show()

        torch.save(model, f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.ckpt')

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
            y_hat = model(torch.tensor(dataset.test_data.reshape(-1, hyper_params["timesteps"],hyper_params["rnn_config"]["ninp"]), dtype=torch.float32).to(device))
            y_hat_train = model(torch.tensor(dataset.train_data.reshape(-1, hyper_params["timesteps"],hyper_params["rnn_config"]["ninp"]), dtype=torch.float32).to(device))
        elif hyper_params['model'] == 'dnn':
            y_hat = model(torch.tensor(dataset.test_data, dtype=torch.float32).to(device))
            y_hat_train = model(torch.tensor(dataset.train_data, dtype=torch.float32).to(device))

        if hyper_params['loss'] == 'ce':
            labels = torch.tensor(dataset.test_label,dtype=torch.long).reshape([-1]).to(device)
            labels_train = torch.tensor(dataset.train_label,dtype=torch.long).reshape([-1]).to(device)
            test_acc = accuracy_softmax(labels, y_hat)
            train_acc = accuracy_softmax(labels_train, y_hat_train)
        elif hyper_params['loss'] == 'mse':
            labels = torch.tensor(np.eye(np.max(dataset.test_label) - np.min(dataset.test_label) + 1)[dataset.test_label]).to(device)
            labels = torch.tensor(labels, dtype=torch.float).reshape([-1, 2]).to(device)
            test_acc = accuracy_softmax_mse(labels, y_hat)

        #accuracy (torch.max(torch.softmax(y_hat, dim=1),dim=1)[1] == labels).float().mean()
        print('Test 1 Accuracy {} | Confidence {}'.format(100 * test_acc,confidence_score(torch.softmax(y_hat, dim=1).cpu().detach().numpy(), dataset.test_label)))
        print('Train 1 Accuracy {} | Confidence {}'.format(100 * train_acc,confidence_score(torch.softmax(y_hat_train, dim=1).cpu().detach().numpy(), dataset.train_label)))
        # print('Test 2 Accuracy {} | Confidence {}'.format(100 * test_acc,accuracy_softmax(dataset.test_label,torch.softmax(y_hat, dim=1))))


        # tc2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],dtype=torch.float32)
        # tc2 = torch.tensor([[100, 2, 2, 2, 2, 2, 2, 2, 2, 2]],dtype=torch.float32)
        # tc2 = torch.tensor([[1, 0, 0, 0, 0, 0, 1, 1, 1, 1]],dtype=torch.float32)
        # o=model(tc2)
        # print(o)
        # print(np.argsort(o.cpu().data.numpy()))
        # print(torch.softmax(o,dim=1))
        # print(torch.max(torch.softmax(o,dim=1),1))
        # Save the model checkpoint
        # torch.save(model, f'{hyper_params["model_save_path"]}{hyper_params["model_name"]}.ckpt')
        return model


if __name__ == '__main__':

    # for debugging...
    # import debugpy
    # os. chdir("/work/rdoddaiah/TimeSeriesSaliencyMaps/nte")

    # use_model = 'rnn'
    use_model = 'cnn'
    use_loss = 'ce'
    # use_loss = 'mse'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = BurstExistence10()
    # dataset = BurstLocation10()
    # dataset = BurstFrequency10()
    # dataset = BurstStrength10()
    # dataset = BurstTimeDifferenceExistence10()
    # dataset = BurstTimeDifferenceStrength10()

    # device = 'cpu' if torch.cuda.is_available() else 'cpu'

    # dataset = EOGHorizontalSignalDataset_MultiClass()
    dataset = ACSFDataset_MultiClass()
    # dataset = AbnormalHeartbeatDataset_MultiClass()
    # dataset = PlaneDataset_MultiClass()
    # dataset = TraceDataset_MultiClass()
    # dataset = MeatDataset_MultiClass()
    # dataset = ECG5000Dataset_MultiClass()
    # dataset = RockDataset_MultiClass()
    # dataset = SmallKitchenAppliancesDataset_MultiClass()
    # dataset = CricketXDataset_MultiClass()
    # dataset = BlipMCDataset()
    model_name = f"{dataset.name}_mc_{use_model}_{use_loss}_mc"

    num_class = dataset.num_classes
    print(f"dataset.num_classes {num_class}")

    TIMESTEPS = len(dataset.train_data[0])
    # Hyper-parameters
    hyper_params = Munch({
        "model_save_path": "/tmp/",
        "model": use_model,
        "loss": use_loss,
        "model_name": model_name,
        "dependency_meta": "",
        "timesteps": TIMESTEPS,
        "num_classes": num_class,
        "rnn_config": {
            "ninp": 1,
            "nhid": 75,
            "nlayers": 3,
            "nclasses": num_class
        },
        "dnn_config": {
            "layers": [2100, 900, 300, num_class],
        },
        "batch_size": 32,
        "num_epochs": 500,
        "learning_rate": 1e-3
        # blipmc
        # "batch_size": 32,
        # "num_epochs": 200,
        # "learning_rate": 1e-3
        # "num_epochs": 50,
        # "learning_rate": 1e-4
    })

    TIMESTEPS = len(dataset.train_data[0])
    data = dataset.train_data[10]

    print("Data Loaded . . . ")
    print(f"Train -  Data: {dataset.train_data.shape} | Label: {dataset.train_label.shape}")
    print(f"Test -  Data: {dataset.test_data.shape} | Label: {dataset.test_data.shape}")
    # nte_plots = 0

    # visualize the target variable
    # g = sns.countplot(dataset.train_label)
    # g.set_xticklabels(['Train Class 0 ', 'Train Class 1'])
    # plt.show()
    # g = sns.countplot(dataset.test_label)
    # g.set_xticklabels(['Test Class 0 ', 'Test Class 1'])
    # plt.show()
    train_black_box_model_multi_class(dataset=dataset, hyper_params=hyper_params)
