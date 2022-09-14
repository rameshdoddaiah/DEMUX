# -*- coding: utf-8 -*-
"""
| **@created on:** 9/28/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
"""
import torch
import cv2
import sys
import numpy as np
import os
import json
import wandb
import pandas as pd
import ssl
from nte.experiment.utils import get_image, tv_norm, \
    save, numpy_to_torch, load_model, get_model, send_plt_to_wandb, save_timeseries, dataset_mapper, \
    backgroud_data_configuration, get_run_configuration
from nte.experiment.evaluation import qm_plot
import tqdm
import shortuuid
import matplotlib.pyplot as plt
from nte.models.saliency_model import SHAPSaliency, LimeSaliency
from nte.models.saliency_model.rise_saliency import RiseSaliency
from nte.models.saliency_model.cm_gradient_saliency import CMGradientSaliency
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import median_filter
import logging
import random
from nte.experiment.evaluation import run_evaluation_metrics
import seaborn as sns
from nte.utils.perturbation_manager import PerturbationManager
from nte.experiment.softdtw_loss_v1 import SoftDTW
from nte.models.saliency_model import Saliency
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import math
from nte.utils.priority_buffer import PrioritizedBuffer
# %matplotlib inline
from sklearn.manifold import TSNE

  
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
import torch.optim as optim

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler

class LSTMGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear0 = nn.Linear(in_dim, hidden_dim)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        # self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        # recurrent_features, _ = self.linear0(input,self.hidden_dim)
        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs


class LSTMDiscriminator(nn.Module):
    def __init__(self, in_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 10), nn.Sigmoid())

    def forward(self, input):
        # batch_size, seq_len = input.size(0), input.size(1)
        batch_size, seq_len = 1, input.shape[0]
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, 10)
        return outputs



class PertMultiClassSaliency(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(PertMultiClassSaliency, self).__init__(background_data=background_data,
                                             background_label=background_label,
                                             predict_fn=predict_fn)
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.r_index = random.randrange(0, len(self.background_data)) if self.args.r_index < 0 else self.args.r_index
        self.rs_priority_buffer = None
        self.ro_priority_buffer = None
        self.eps = 1.0
        self.eps_decay = 0.9991#0.9996#0.997

    def save_perturbations_original_timeseries(self, kwargs, various_perturbations,various_perturbations_labels, various_perturbations_masks,orig_first_category,orig_second_category,orig_third_category,reached_perturbs,outputs):
        CRED = '\033[91m'
        CEND = '\033[0m'
        CGRN = '\033[92m'
        CGRN = '\033[92m'
        CCYN = "\033[36m"
        plt.clf()
        plt.figure()
        various_perturbations_mean = np.mean(various_perturbations, axis=0)
        various_perturbations_masks_mean = np.mean(various_perturbations_masks, axis=0)
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=6)
        # print(CRED+f"various_perturbations:{np.array(various_perturbations)}"+CEND)
        # print(CGRN+f"various_perturbations_mean:{various_perturbations_mean}"+CEND)
        # print(CRED+f"various_perturbations_masks:{np.array(various_perturbations_masks)}"+CEND)
        # print(CGRN+f"various_perturbations_masks_mean:{various_perturbations_masks_mean}"+CEND)
        various_perturbs_pred_latest = self.softmax_fn(self.predict_fn(torch.tensor(various_perturbations[-1], dtype=torch.float32).reshape([1, -1])))
        various_perturbs_pred = self.softmax_fn(self.predict_fn(torch.tensor(various_perturbations_mean, dtype=torch.float32).reshape([1, -1])))
        print(CRED+f"Various Perturbations Mean Predictions: {various_perturbs_pred}")
        print(CRED+f"Various Perturbations Mean : {various_perturbations_mean}")
        print(CRED+f"Various Perturbations Variance : {np.var(various_perturbations, axis=0)}")
        print(f"outputs Orig | I : {orig_first_category} {outputs[orig_first_category]:.4f}| II: {orig_second_category} {outputs[orig_second_category]:.4f}| III:{orig_third_category} {outputs[orig_third_category]:.4f}")
        print(f"various Mean | I : {orig_first_category} {various_perturbs_pred[0][orig_first_category]:.4f}| II: {orig_second_category} {various_perturbs_pred[0][orig_second_category]:.4f}| III:{orig_third_category} {various_perturbs_pred[0][orig_third_category]:.4f}")
        print(f"various Late | I : {orig_first_category} {various_perturbs_pred_latest[0][orig_first_category]:.4f}| II: {orig_second_category} {various_perturbs_pred_latest[0][orig_second_category]:.4f}| III:{orig_third_category} {various_perturbs_pred_latest[0][orig_third_category]:.4f}")
        plt.xlabel(str(reached_perturbs)+'X_LABEL')
        plt.plot(various_perturbations_mean,label="Various Signal AVG")
        plt.plot(various_perturbations_masks_mean[0][0], label="Various Signal Masks AVG")
        mask_t = various_perturbations_masks_mean[0][0]
        abs_mask_t = mask_t  # np.abs(mask)
        mask_min_t = np.min(abs_mask_t)
        mask_max_t = np.max(abs_mask_t)
        norm_mask_t = (mask_t - mask_min_t) / (mask_max_t - mask_min_t)
        plt.plot(norm_mask_t, label="Normalized Various Signal Masks AVG")
        plt.show()
        plt.savefig(str(reached_perturbs) + 'avg_various_plot.png')
        plt.clf()
        plt.figure()
        plt.xlabel('dataset mean')
        if self.args.first_class == 'yes':
            class_mean_t = orig_first_category
        elif self.args.second_class == 'yes':
            class_mean_t = orig_second_category
        elif self.args.third_class == 'yes':
            class_mean_t = orig_third_category
        else:
            class_mean_t = label


    #     nz =100
    #     seq_len = various_perturbations_mean.shape[0]
    #     in_dim = nz+1

    #     device = torch.device("cpu")
    #     netD = LSTMDiscriminator(in_dim=seq_len, hidden_dim=256).to(device)
    #     netG = LSTMGenerator(in_dim=seq_len, out_dim=seq_len, hidden_dim=256).to(device)

    #     print("|Generator Architecture|\n", netG)
    #     print("|Discriminator Architecture|\n", netD)

    #     fixed_noise = torch.randn(1, seq_len, nz, device=device)
    #     # deltas = dataset.sample_deltas(opt.batchSize).unsqueeze(2).repeat(1, seq_len, 1)
    #     # fixed_noise = torch.cat((fixed_noise, deltas), dim=2)
    #     fixed_noise = torch.cat((fixed_noise, fixed_noise), dim=2)

    #     real_label = 1
    #     fake_label = 0
        
    #     optimizerD = optim.Adam(netD.parameters(), lr=0.002)
    #     optimizerG = optim.Adam(netG.parameters(), lr=0.002)

    #     criterion = nn.BCELoss().to(device)
    #     delta_criterion = nn.MSELoss().to(device)
    #     mse_loss_fn = torch.nn.MSELoss(reduction='mean')
    #     loss_ce = nn.CrossEntropyLoss()

    #     total_epochs = 10
    #     for epoch in range(total_epochs):
    #         for i, data in enumerate(zip(various_perturbations, various_perturbations_labels)):
    #             niter = epoch * len(various_perturbations_labels) + i
        
    #             #Save just first batch of real data for displaying
    #             # if i == 0:
    #             #     real_display = data[0]
            
    #             ############################
    #             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    #             ###########################

    #             #Train with real data
    #             # netD.zero_grad()
    #             real = data
    #             batch_size, seq_len = 1,len(real[0])
    #             real_label = data[1]
    #             # label = torch.full((batch_size, seq_len, 1), real_label, device=device)
                
    #             # deltas = (real[:, -1] - real[:, 0]).unsqueeze(2).repeat(1, seq_len, 1)
    #             # real = torch.cat((real, deltas), dim=2)
    #             real = torch.tensor(data[0],dtype=torch.float32)
    #             # outputs_sfn = self.softmax_fn(self.predict_fn(real.reshape([1, -1])))[0]
    #             # label = torch.argmax(outputs_sfn)
    #             # output_pred = torch.max(outputs_sfn)
    #             real=real.unsqueeze(0).unsqueeze(0)
    #             # real = torch.cat((real, real), dim=0)
    #             # output_d = netD(real)
    #             # errD_real = criterion(output, label)
    #             # errD_real.backward()
    #             # D_x = output.mean().item()
                
    #             # #Train with fake data
    #             # noise = torch.randn(batch_size, seq_len, nz, device=device)
    #             # #Sample a delta for each batch and concatenate to the noise for each timestep
    #             # deltas = dataset.sample_deltas(batch_size).unsqueeze(2).repeat(1, seq_len, 1)
    #             # noise = torch.cat((noise, deltas), dim=2)

    #             orig_label =  data[1]
    #             netG.zero_grad()
    #             optimizerG.zero_grad()
    #             output_g = netG(real)
    #             # loss  = mse_loss_fn(outputs_sfn[orig_label], kwargs['target'].squeeze(0)[category])

    #             # outputs_sfn = self.softmax_fn(self.predict_fn(real.reshape([1, -1])))[0]
    #             # label = torch.argmax(outputs_sfn)
    #             # output_pred = torch.max(outputs_sfn)

    #             # label.fill_(real_label)
    #             # output = netD(torch.cat((fake.detach(), deltas), dim=2))
    #             # errD_fake = criterion(output_g, label)
    #             # loss = loss_ce(output_g.squeeze(0), torch.tensor([orig_label],dtype=torch.long))

    #             desired_label = 2
    #             loss = loss_ce(output_g.squeeze(0), torch.tensor([desired_label],dtype=torch.long))
    #             loss.backward()
    #             # D_G_z1 = output.mean().item()
    #             # errD = errD_real + errD_fake
    #             # optimizerD.step()
                
    #             #Visualize discriminator gradients
    #             # for name, param in netD.named_parameters():
    #             #     writer.add_histogram("DiscriminatorGradients/{}".format(name), param.grad, niter)

    #             ############################
    #             # (2) Update G network: maximize log(D(G(z)))
    #             ###########################
    #             # netG.zero_grad()
    #             # label.fill_(real_label) 
    #             # output = netD(torch.cat((fake, deltas), dim=2))
    #             # errG = criterion(output, label)
    #             # errG.backward()
    #             # D_G_z2 = output.mean().item()
                
    #             optimizerG.step()
    #             # netG.data.clamp_(0, 1)
                
    #             #Visualize generator gradients
    #             # for name, param in netG.named_parameters():
    #             #     writer.add_histogram("GeneratorGradients/{}".format(name), param.grad, niter)
                
    #             ###########################
    #             # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
    #             ###########################

    #             #Report metrics
    #             # print('[%d][%d] Loss_Gen: %.4f' % (epoch, i, loss))
    #             print(f'Epoch:{epoch} i:{i} loss:{loss} output_g:{output_g.data}')
    #             # writer.add_scalar('DiscriminatorLoss', errD.item(), niter)
    #             # writer.add_scalar('GeneratorLoss', errG.item(), niter)
    #             # writer.add_scalar('D of X', D_x, niter) 
    #             # writer.add_scalar('D of G of z', D_G_z1, niter)

    # #             nn.linear() stack of FC layers
    # #             loss CE

    # #             1,2,3,4,5,6 TS
    # #          CAT   1,0,0,1,1,1 // input time series
    # #          CAT   1,0,0,1,1,1 // input time series
    # #                1,0,0,1,1,1 // input time series
    # #                0.1,0.2,0,1,1.4,1 // Seed time series generator
    # #   desired DOG1 1,1,10,0,// output
    # #                B1,B2,B3 // attention 
                   

    # #               orig_label is cat
    # #               output_m =  generator_model(input time series)
    # #               yhat_bbm = bbm(output_m time series)
    # #               loss = loss_ce(yhat_bbm, desired(dog)_label)
    # #               delta = input_time_series-output_ml

    # #               total_loss = loss + delta
    # #               total_loss.backprop




    #     for i, data in enumerate(zip(various_perturbations, various_perturbations_labels)):
    #         real = torch.tensor(data[0],dtype=torch.float32)
    #         outputs_sfn = self.softmax_fn(self.predict_fn(real.reshape([1, -1])))[0]
    #         label = torch.argmax(outputs_sfn)
    #         output_pred = torch.max(outputs_sfn)
    #         real=real.unsqueeze(0).unsqueeze(0)
    #         fake = netG(real)
    #         fake_output = self.softmax_fn(self.predict_fn(fake.reshape([1, -1])))[0]
    #         print(f'FAKE...{fake} desired_label {desired_label} label {label} output_pred {output_pred} real {real} fakeoutput softmax {fake_output}')
    #         break



        plt.clf()
        plt.figure()
        class_mean_data = [getattr(self.args.dataset, f"test_class_{class_mean_t}_mean")]
        plt.plot(class_mean_data[0],label="Various Signal AVG")
        plt.show()
        plt.savefig('class_mean_data_plot.png')
        plt.clf()
        plt.figure()
        class_mean_data_1 = [getattr(self.args.dataset, f"test_class_{orig_first_category}_mean")]
        plt.plot(
        class_mean_data_1[0], label="Various Signal AVG")
        plt.show()
        plt.savefig('First_class_mean .png')
        plt.clf()
        plt.figure()
        class_mean_data_1 = getattr(self.args.dataset, f"test_class_{orig_first_category}_data")
        class_indices_first = getattr(self.args.dataset, f"test_class_{orig_first_category}_indices")
        for i in range(len(class_indices_first)):
            plt.plot(class_mean_data_1[i], label="Various Signal AVG")
        plt.show()
        plt.savefig('First_class_1 .png')
        plt.clf()
        plt.figure()
        class_mean_data_1 = [getattr(self.args.dataset, f"test_class_{orig_second_category}_mean")]
        plt.plot(class_mean_data_1[0], label="Various Signal AVG")
        plt.show()
        plt.savefig('Second_class_mean .png')
        plt.clf()
        plt.figure()
        class_mean_data_1 = getattr(self.args.dataset, f"test_class_{orig_second_category}_data")
        class_indices_second = getattr(self.args.dataset, f"test_class_{orig_second_category}_indices")
        for i in range(len(class_indices_second)):
            plt.plot(class_mean_data_1[i], label="Various Signal AVG")
            if i == 10:
                break
        plt.show()
        plt.savefig('Second_class_1 .png')
        plt.clf()
        plt.figure()
        class_mean_data_1 = [getattr(self.args.dataset, f"test_class_{orig_third_category}_mean")]
        plt.plot(class_mean_data_1[0], label="Various Signal AVG")
        plt.show()
        plt.savefig('Third_class_mean .png')
        plt.clf()
        plt.figure()
        class_mean_data_1 = getattr(self.args.dataset, f"test_class_{orig_third_category}_data")
        class_indices_third = getattr(self.args.dataset, f"test_class_{orig_third_category}_indices")
        for i in range(len(class_indices_third)):
            plt.plot(class_mean_data_1[i], label="Various Signal AVG")
            if i == 10:
                break
        plt.show()
        plt.savefig('Third_class_1.png')
        plt.clf()
        plt.figure()
        print(f"11:np.unique {np.unique(various_perturbations_labels)} len {len(np.unique(various_perturbations_labels))}")
        from mpl_toolkits.mplot3d import Axes3D
        tsne = TSNE(n_components=2, random_state=0)
        data_X = various_perturbations[:reached_perturbs]
        tsne_obj= tsne.fit_transform(data_X)
        tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1]})
        tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'Labels':various_perturbations_labels})
        num_classes = getattr(kwargs['dataset'],f"num_classes")
        # sns.scatterplot(x="X", y="Y",data=tsne_df)
        # # sns.scatterplot(x="X", y="Y",hue="Labels",palette=['purple','red','orange','cyan','blue','dodgerblue','green','lightgreen','darkcyan', 'black','cyan','yellow'],legend='full',data=tsne_df)
        if num_classes == 5:
            sns.scatterplot(x="X", y="Y",hue="Labels",palette=['purple','red','orange','cyan','blue'],legend='full',data=tsne_df)
        else:
            sns.scatterplot(x="X", y="Y",hue="Labels",palette=['purple','red','orange','cyan','blue','dodgerblue','green','lightgreen','darkcyan','black'],legend='full',data=tsne_df)
            # sns.scatterplot(x="X", y="Y",hue="Labels",palette='colorblind',legend='full',data=tsne_df)
            # sns.scatterplot(x="X", y="Y",hue="Labels",palette=['purple','red','orange','cyan','blue','dodgerblue','green','lightgreen','darkcyan', 'black','cyan','yellow'],legend='full',data=tsne_df)

        # tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'Z':tsne_obj[:,2]})
        # tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'Z':tsne_obj[:,2],'Labels':various_perturbations_labels})
        # sns.scatterplot(x="X", y="Y",data=tsne_df)
        # sns.scatterplot(x="X", y="Y",z='Z',hue="Labels",palette=['purple','red','orange','cyan','blue','dodgerblue','green','lightgreen','darkcyan', 'black','cyan','yellow'],legend='full',data=tsne_df)
        # sns.scatterplot(x="X", y="Y",z='Z', hue="Labels",palette=['purple','red','orange','cyan','blue'],legend='full',data=tsne_df)
        # sns.scatterplot("X", "Y",'Z', hue="Labels",palette=['purple','red','orange','cyan','blue'],legend='full',data=tsne_df)

        # sns.set(style = "darkgrid")
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        # x = tsne_df['X']
        # y = tsne_df['Y']
        # z = tsne_df['Z']

        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")

        # # ax.scatter(x, y, z)
        # ax.scatter(xs=x,ys=y,zs=z,c = tsne_df['Labels'], cmap = plt.cm.get_cmap("nipy_spectral_r",10))

        # plt.show()
        # plt.savefig('TSNE_Plots_3D.png')

        # plt.clf()
        # plt.figure()
        # sns.scatterplot(x="X", y="Y",hue="Labels",palette=['purple','red','orange','cyan','blue'],legend='full',data=tsne_df)
        plt.show()
        plt.savefig('TSNE_Plots_2D.png')
        plt.clf()
        plt.figure()

        all_data_set =  []
        all_data_set_label =  []
        num_classes = getattr(kwargs['dataset'],f"num_classes")
        for class_i in range(num_classes):
            indices = getattr(self.args.dataset, f"test_class_{class_i}_indices")
            class_i_data = getattr(self.args.dataset, f"test_class_{class_i}_data")
            for ind in range(len(indices)):
                all_data_set.append(class_i_data[ind])
                all_data_set_label.append(class_i)
        tsne_all_data = TSNE(n_components=2, random_state=0)
        tsne_obj_all_data = tsne_all_data.fit_transform(all_data_set)
        tsne_df_all_data = pd.DataFrame({'X':tsne_obj_all_data[:,0],'Y':tsne_obj_all_data[:,1],'all_labels':all_data_set_label})

        print(f"22:np.unique {np.unique(all_data_set_label)} len {len(np.unique(all_data_set_label))}")
        # sns.scatterplot(x="X", y="Y",data=tsne_df)
        # # sns.scatterplot(x="X", y="Y",hue="Labels",palette=['purple','red','orange','cyan','blue','dodgerblue','green','lightgreen','darkcyan', 'black','cyan','yellow'],legend='full',data=tsne_df)
        if num_classes == 5:
            sns.scatterplot(x="X", y="Y",hue="all_labels",palette=['purple','red','orange','cyan','blue'],legend='full',data=tsne_df_all_data)
        else:
            # sns.scatterplot(x="X", y="Y",hue="Labels",palette=['purple','red','orange','cyan','blue','green','yellow'],legend='full',data=tsne_df)
            # sns.scatterplot(x="X", y="Y",hue="all_labels",palette=['purple','red','orange','cyan','blue','green','yellow'],legend='full',data=tsne_df_all_data)
            # sns.scatterplot(x="X", y="Y",hue="all_labels",palette='colorblind',legend='full',data=tsne_df_all_data)
            # sns.scatterplot(x="X", y="Y",hue="Labels",palette='colorblind',legend='full',data=tsne_df)
            sns.scatterplot(x="X", y="Y",hue="all_labels",palette=['purple','red','orange','cyan','blue','dodgerblue','green','lightgreen','darkcyan', 'black','cyan','yellow'],legend='full',data=tsne_df_all_data)
        plt.show()
        plt.savefig('TSNE_Plots_all_data_2D.png')


        v_perturbs = []
        v_perturbs_labels = []
        v_label = class_mean_t
        for v in range(len(various_perturbations_labels)):
            if various_perturbations_labels[v] == v_label:
                v_perturbs.append(various_perturbations[v])
                v_perturbs_labels.append(various_perturbations_labels[v])

        print(CRED+f"Class 2 Various Perturbations Mean : {np.mean(v_perturbs, axis=0)}")
        print(CRED+f"Class 2 Various Perturbations Variance : {np.var(v_perturbs, axis=0)}")
        plt.clf()
        plt.figure()
        plt.plot(np.mean(v_perturbs,axis=0), label="Class Mean ")
        plt.plot(np.var(v_perturbs,axis=0), label="Class Variance")
        plt.show()
        plt.savefig('Class_Mean_Variance.png')

        plt.clf()
        plt.figure()
        plt.plot(v_perturbs, label="Class Perturbations")
        plt.show()
        plt.savefig('Class_perturbations.png')

        # from sklearn.linear_model import LinearRegression
        # model_lr = LinearRegression()
        # model_lr.fit(various_perturbations, various_perturbations_labels)
        # r_sq = model_lr.score(various_perturbations, various_perturbations_labels)
        # print('coefficient of determination:', r_sq)
        # print('intercept:', model_lr.intercept_)
        # print('slope:', model_lr.coef_)
        # print('various mean :', various_perturbations_mean, sep='\n')
        # print('various clas 2 mean :', v_perturbs, sep='\n')
        # y_pred = model_lr.predict([various_perturbations_mean])
        # print('predicted response:', y_pred, sep='\n')
        # y_pred = model_lr.predict([np.mean(v_perturbs,axis=0)])
        # print('predicted response for class 2:', y_pred, sep='\n')



    def priority_dual_greedy_pick_rt(self, kwargs, data, label):
        self.eps *= self.eps_decay
        if np.random.uniform() < self.eps:
            self.mode = 'Explore'
            # todo: Change Rs and Ro dynamically
            num_classes = getattr(kwargs['dataset'],f"num_classes")
            rs_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
            Rs, rs_weight = [getattr(self.args.dataset, f"test_class_{int(label)}_data")[rs_index[0]]], [1.0]

            if num_classes >2:

                min_class = np.argmin(kwargs['target'].cpu().data.numpy())
                max_class = np.argmax(kwargs['target'].cpu().data.numpy())
                orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy())
                orig_first_category = orig_sorted_category[-1]
                orig_second_category = orig_sorted_category[-2]
                orig_third_category = orig_sorted_category[-3]
                # orig_fourth_category = orig_sorted_category[-4]
                # label_category_pred_prob = kwargs['target'].cpu().data.numpy()[label]
                # first_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_first_category]
                # second_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_second_category]
                # third_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_third_category]
                # fourth_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_fourth_category]

                if self.args.grad_replacement == 'random_instance':
                    rand_choice = list(range(0,int(label))) + list(range(int(label)+1, num_classes))
                    rand_label = np.random.choice(rand_choice)
                    gen_index =  len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{rand_label}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'min_class_random_instance':
                    gen_index =  len(getattr(self.args.dataset,f"test_class_{min_class}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{min_class}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'max_class_random_instance':
                    gen_index =  len(getattr(self.args.dataset,f"test_class_{max_class}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{max_class}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'next_class_random_instance':

                    if self.args.first_class == 'yes':
                        next_class = orig_second_category
                    elif self.args.second_class == 'yes':
                        next_class = orig_third_category
                    # elif self.args.third_class == 'yes':
                    #     next_class = orig_fourth_category
                    else:
                        next_class = min_class

                    gen_index =  len(getattr(self.args.dataset,f"test_class_{next_class}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{next_class}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'min_class_mean':
                    ro_index = 0
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{next_class}_mean")], [1.0]
                else:
                    ro_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{1-int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{1-int(label)}_data")[ro_index[0]]], [1.0]

            else:
                ro_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{1-int(label)}_data")))]
                Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{1-int(label)}_data")[ro_index[0]]], [1.0]

        else:
            self.mode = 'Exploit'
            Rs, rs_weight, rs_index = self.rs_priority_buffer.sample(1)
            Ro, ro_weight, ro_index = self.ro_priority_buffer.sample(1)
        return {'rs': [Rs, rs_weight, rs_index],
                'ro': [Ro, ro_weight, ro_index]}

    def dynamic_dual_pick_zt(self, kwargs, data, label):
        ds = kwargs['dataset'].test_class_0_indices
        ls = len(ds)
        ods = kwargs['dataset'].test_class_1_indices
        ols = len(ods)
        self.r_index = random.randrange(0, ls)
        self.ro_index = random.randrange(0, ols)
        Zt = self.background_data[ds[self.r_index]]
        ZOt = self.background_data[ods[self.ro_index]]
        return Zt, ZOt

    def dynamic_pick_zt(self, kwargs, data, label):
        self.r_index = None
        if self.args.grad_replacement == 'zeros':
            Zt = torch.zeros_like(data)
        else:
            if self.args.grad_replacement == 'class_mean':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_class_0_mean, dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_class_1_mean, dtype=torch.float32)
            elif self.args.grad_replacement == 'instance_mean':
                Zt = torch.mean(data).cpu().detach().numpy()
                Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
            elif self.args.grad_replacement == 'random_instance':
                self.r_index = random.randrange(0, len(self.background_data))
                Zt = torch.tensor(self.background_data[self.r_index],
                                  dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    sds = kwargs['dataset'].test_class_0_indices
                    sls = len(sds)
                else:
                    sds = kwargs['dataset'].test_class_1_indices
                    sls = len(sds)
                self.r_index = random.randrange(0, sls)
                Zt = torch.tensor(sds[self.r_index], dtype=torch.float32)
        return Zt

    def static_pick_zt(self, kwargs, data, label):
        if self.args.grad_replacement == 'zeros':
            Zt = torch.zeros_like(data)
        else:
            if self.args.grad_replacement == 'class_mean':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_class_0_mean, dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_class_1_mean, dtype=torch.float32)
            elif self.args.grad_replacement == 'instance_mean':
                Zt = torch.mean(data).cpu().detach().numpy()
                Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
            elif self.args.grad_replacement == 'random_instance':
                Zt = torch.tensor(self.background_data[self.r_index], dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][0],
                                      dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][1],
                                      dtype=torch.float32)
        return Zt

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)

    def generate_saliency(self, data, label, **kwargs):

        # if self.args.enable_seed:
        #     set_global_seed(self.args.seed_value)
        # todo: Change to dynamic category

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)


        # p=torch.tensor([0.1000, 0.6000, 0.1000, 0.1000, 1.0000, 0.1000, 0.1000, 0.1000, 0.1000,0.1000],dtype=torch.float32)
        # data = torch.tensor([0.1000, 0.6000, 0.1000, 0.1000, 1.0000, 0.1000, 0.1000, 0.1000, 0.1000,0.1000],dtype=torch.float32)
        # data = data +0.00001
        reached_perturbs = 1
        orig_label = label
        flag = 0
        orig_pred_prob = (kwargs['target'].cpu().data.numpy())
        category = np.argmax(kwargs['target'].cpu().data.numpy())
        orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy())
        ordered_orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy()*-1)
        orig_first_category = orig_sorted_category[-1]
        orig_second_category = orig_sorted_category[-2]
        orig_third_category = orig_sorted_category[-3]
        # orig_fourth_category = orig_sorted_category[-4]
        label_category_pred_prob = kwargs['target'].cpu().data.numpy()[label]
        first_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_first_category]
        second_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_second_category]
        third_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_third_category]
        # fourth_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_fourth_category]

        print(f'data {data}')
        print(f"kwargs target Probs {(kwargs['target'].cpu().data.numpy())}")
        print(f"ordered_orig_category={ordered_orig_sorted_category}")
        print(f"orig_first_category  ={orig_first_category}, pred {first_category_pred_prob}")
        print(f"orig_second_category ={orig_second_category}, pred {second_category_pred_prob}")
        print(f"orig_third_category  ={orig_third_category},pred {third_category_pred_prob}")
        # print(f"orig_fourth_category ={orig_fourth_category},pred {fourth_category_pred_prob}")
        print(f"label{label} label_category{label_category_pred_prob}")

        # if self.args.first_class == 'yes':
        #     label = orig_first_category
        # elif self.args.second_class == 'yes':
        #     label = orig_second_category
        # elif self.args.third_class == 'yes':
        #     label = orig_third_category
        # else:
        #     label = label

        top_n_masks = []
        top_n_various_perturbations = []
        top_n_norm_masks = []
        top_n_prob_norm_masks = []

        num_classes = getattr(kwargs['dataset'], f"num_classes")

        if (self.args.top_n == 'yes'):
            # top_n_range = 3
            top_n_range = num_classes
        else:
            top_n_range = 1

        for top_n in range(top_n_range):

            # if (top_n <3):
            #     continue

            various_perturbations = []
            various_perturbations_labels = []
            various_perturbations_masks = []
            reached_perturbs = 0

            # rand_label = np.random.randint(0,num_classes)
            rand_choice = list(range(0, int(label))) + list(range(int(label)+1, num_classes))
            rand_label = np.random.choice(rand_choice)
            self.rs_priority_buffer = PrioritizedBuffer(background_data=getattr(kwargs['dataset'], f"test_class_{int(label)}_data"))
            if num_classes > 2:
                self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(kwargs['dataset'], f"test_class_{rand_label}_data"))
            else:
                self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(kwargs['dataset'], f"test_class_{1 - int(label)}_data"))

            self.eps = 1.0

            # Calculate Gradient Replacement
            # if self.args.grad_replacement == 'class':
            #     pass
            # elif self.args.grad_replacement == 'instance':
            #     Zt = torch.mean(data).cpu().detach().numpy()
            #     Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
            # elif self.args.grad_replacement == 'random':
            #     pass
            # elif self.args.grad_replacement == 'random_class':
            #     original_class_predictions = torch.max(kwargs['target']).item()

            if kwargs['save_perturbations']:
                self.perturbation_manager = PerturbationManager(original_signal=data.cpu().detach().numpy().flatten(),algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()),
                    original_label=label, sample_id=self.args.single_sample_id)

            plt.plot(data, label="Original Signal Norm")
            gkernel = cv2.getGaussianKernel(3, 0.5)
            gaussian_blur_signal = cv2.filter2D(data.cpu().detach().numpy(), -1, gkernel).flatten()
            plt.plot(gaussian_blur_signal, label="Gaussian Blur")
            median_blur_signal = median_filter(data, 3)
            plt.plot(median_blur_signal, label="Median Blur")
            blurred_signal = (gaussian_blur_signal + median_blur_signal) / 2
            plt.plot(blurred_signal, label="Blurred Signal")
            mask_init = np.random.uniform(size=len(data), low=-1e-2, high=1e-2)
            #TODO for args.second_class, args.third_class onwards mask_init should be proportional or normalized
            # mask_init = np.full(shape=len(data), fill_value=1e-2)
            blurred_signal_norm = blurred_signal / np.max(blurred_signal)
            plt.plot(blurred_signal_norm, label="Blur Norm")

            # to log noise

            # if self.args.enable_noise:
            #     noise = np.zeros(original_signal.shape, dtype=np.float32)
            #     cv2.randn(noise, 0, 0.2)
            #     log_noise = original_signal
            #     log_noise+= noise
            #     log_noise = log_noise/np.max(log_noise)
            #     plt.plot(log_noise, label="Orig_Noise Norm")

            if self.enable_wandb:
                wandb.log({'Initialization': plt}, step=0)

            # # visualize the class imbalance plots if needed
            # g = sns.countplot(dataset.train_label)
            # g.set_xticklabels(['Train Class 0 ', 'Train Class 1'])
            # plt.show()

            # if ENABLE_WANDB:
            #     wandb.log({'Train Class Balanced vs Imbalanced': plt})

            # g = sns.countplot(dataset.test_label)
            # g.set_xticklabels(['Test Class 0 ', 'Test Class 1'])
            # plt.show()

            # if ENABLE_WANDB:
            #     wandb.log({'Test Class Balanced vs  Imbalanced': plt})

            blurred_signal = torch.tensor((blurred_signal_norm.reshape(1, -1)), dtype=torch.float32)
            mask = numpy_to_torch(mask_init, use_cuda=self.use_cuda)
            masks_init_ones = mask
            # todo: Ramesh - Upsample?
            # if use_cuda:
            #     upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
            # else:
            #     upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))

            optimizer = torch.optim.Adam([mask], lr=self.args.lr)

            if self.args.enable_lr_decay:
                scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

            print(f"{self.args.algo}: Category with highest probability {category}")
            print(f"{self.args.algo}: Optimizing.. ")

            metrics = {"TV Norm": [], 'TV Coeff': [], "MSE": [], "Budget": [], "Total Loss": [],
                    "Confidence": [],
                    "Saliency Var": [],
                    "Saliency Sum": [], "MSE Coeff": [], "L1 Coeff": [], "Mean Gradient": [],
                    "Category": [],
                    "Label_pred_prob": [],
                    "First_pred_prob": [],
                    "Second_pred_prob": [],
                    "Third_pred_prob": [],
                    "var_pert_len": [],
                    # "DTW": [], "DTW Coeff":[], 'EUC': [],
                    "epoch": {}, 'MSE Var': [], 'DIST': []}

            metrics['Label_pred_prob'].append(float(label_category_pred_prob.item()))
            metrics['First_pred_prob'].append(float(first_category_pred_prob.item()))
            metrics['Second_pred_prob'].append(float(second_category_pred_prob.item()))
            metrics['Third_pred_prob'].append(float(third_category_pred_prob.item()))
            metrics['var_pert_len'].append(float(1))

            if (self.args.ce == 'yes'):
                mse_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            elif(self.args.kl == 'yes'):
                mse_loss_fn = torch.nn.functional.kl_div
                # mse_loss_fn = torch.nn.KLDivLoss()
                # torch.distribution.kl_divergence.
                # mse_loss_fn = torch.nn.functional.kl_div(reduction ='sum')
                # mse_loss_fn = torch.nn.functional.kl_div(Q.log(), P, None, None, 'sum')
            elif(self.args.mse == 'yes'):
                mse_loss_fn = torch.nn.MSELoss(reduction='mean')
            else:
                mse_loss_fn = torch.nn.MSELoss(reduction='mean')

            softdtw_loss_fn = SoftDTW()
            original_class_predictions = torch.max(kwargs['target']).item()

            # Static pick
            # Zt = self.static_pick_zt(kwargs=kwargs, data=data, label=label)

            mask_append = []
            for i in (range(self.args.max_itr)):
                CUR_DIR = kwargs['save_dir'] + f'/epoch_{i}/'
                # os.system(f"mkdir -p {CUR_DIR}")

                picks = self.priority_dual_greedy_pick_rt(kwargs=kwargs, data=data, label=label)
                rs, rs_weight, rs_index = picks['rs']
                ro, ro_weight, ro_index = picks['ro']
                Rt = []
                Rm = []

                for e, (rs_e, ro_e, m) in enumerate(zip(rs[0].flatten(), ro[0].flatten(), mask.detach().numpy().flatten())):
                    if m < 0:
                    #     # Rt.append(data[e])
                    #     # Rt.append(ZOt[e])
                        Rt.append(rs_e)
                        Rm.append("z")
                    else:
                        Rt.append(ro_e)
                    # Rt.append(Zt[e])
                        Rm.append("o")
                Rt = torch.tensor(Rt, dtype=torch.float32)

                # Rt = self.dynamic_pick_zt(kwargs=kwargs, data=data, label=label)

                upsampled_mask = (mask)

                # mask_append.append(mask.data.detach().numpy())

                #Import TODO
                #Import TODO
                #Import TODO
                #Import TODO
                perturbated_input = data.mul(upsampled_mask) + Rt.mul(1 - upsampled_mask)
                # perturbated_input2 = data.mul(upsampled_mask) + Rt.mul(1 - upsampled_mask)
                
                # perturbated_input = data.mul(upsampled_mask) 



                # perturbated_input = data.mul(upsampled_mask)
                # perturbated_input = data.mul(upsampled_mask) + Zt.mul(1 - upsampled_mask)

                if self.args.enable_blur:
                    # perturbation = blurred_signal.mul(neg_upsampled)
                    perturbation = blurred_signal.mul(upsampled_mask)
                    perturbated_input += perturbation
                    # cv2.imwrite(f"{CUR_DIR}/4-perturbation.png", get_image(perturbation.cpu().detach().numpy()))

                if self.args.enable_noise:
                    noise = np.zeros(data.shape, dtype=np.float32)
                    cv2.randn(noise, 0, 0.3)
                    noise = numpy_to_torch(noise, use_cuda=self.use_cuda)
                    perturbated_input += noise
                    # cv2.imwrite(f"{CUR_DIR}/5-noise.png", get_image(noise.cpu().detach().numpy()))
                perturbated_input = perturbated_input.flatten()
                # cv2.imwrite(f"{CUR_DIR}/6-perturbed_input+noise.png",
                # get_image(perturbated_input.cpu().detach().numpy()))
                # with torch.no_grad():
                if (self.args.ce == 'yes'):
                    outputs = (self.predict_fn(perturbated_input.reshape([1, -1])))
                    metrics['Confidence'].append(float(outputs[0][category].item()))
                    if (self.args.ml == 'yes'):
                        sorted_category = np.argsort(kwargs['target'].cpu().data.numpy())
                        first_category = sorted_category[-1]
                        second_category = sorted_category[-2]
                        third_category = sorted_category[-3]
                elif (self.args.mse == 'yes'):
                    # outputs 1x12
                    if self.args.bbmg == 'yes':
                        timesteps = int(perturbated_input.shape[0]/self.args.window_size)
                        outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape(-1,timesteps,self.args.window_size)))[0]
                    else:
                        outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape([1, -1])))[0]
                    metrics['Confidence'].append(float(outputs[label].item()))
                else:
                    outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape([1, -1])))[0]
                    metrics['Confidence'].append(float(outputs[category].item()))

                    #at whatever confidence we need to tune how much
                    #assume first category is 0.65 prediction probability and we want to reach 0.99 prediction probability
                    #using original paper we can compute what amounts to 0.65(A) and new idea is to use 0.99 (B) as mse and shows the difference by normalizing A and B and showing the difference ?

                if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                    self.perturbation_manager.add_perturbation(perturbation=perturbated_input.cpu().detach().numpy().flatten(), step=i, confidence=metrics['Confidence'][-1])
                if self.args.bbm == 'rnn':
                    c1 = self.args.mse_coeff * mse_loss_fn(outputs[category], kwargs['target'].squeeze(0)[category])
                else:
                    if (self.args.ce == 'yes'):
                        if (self.args.ml == 'yes'):
                            cat_outputs = torch.cat([outputs, outputs, outputs], dim=0)
                            c1 = self.args.mse_coeff * mse_loss_fn(cat_outputs, torch.LongTensor([orig_first_category, orig_second_category, orig_third_category]))
                            #c1 = self.args.mse_coeff * mse_loss_fn(cat_outputs, torch.LongTensor([first_category,first_category, first_category]))
                        else:
                            if self.args.first_class == 'yes':
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs, torch.LongTensor([orig_first_category]))
                            elif self.args.second_class == 'yes':
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs, torch.LongTensor([orig_second_category]))
                            elif self.args.third_class == 'yes':
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs, torch.LongTensor([orig_third_category]))
                            else:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs, torch.LongTensor([top_n]))

                    elif(self.args.kl == 'yes'):
                        c1 = self.args.mse_coeff * mse_loss_fn(outputs.log(),kwargs['target'], None, None, 'sum')

                    elif(self.args.mse == 'yes'):
                        if self.args.first_class == 'yes':
                            if self.args.prob_upto > 0:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[orig_first_category], torch.tensor(self.args.prob_upto, dtype=torch.float32))
                            else:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[orig_first_category], kwargs['target'][orig_first_category])
                        elif self.args.second_class == 'yes':
                            if self.args.prob_upto > 0:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[orig_second_category], torch.tensor(self.args.prob_upto, dtype=torch.float32))
                            else:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[orig_second_category], kwargs['target'][orig_second_category])
                        elif self.args.third_class == 'yes':
                            if self.args.prob_upto > 0:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[orig_third_category], torch.tensor(self.args.prob_upto, dtype=torch.float32))
                            else:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[orig_third_category], kwargs['target'][orig_third_category])
                        else:
                            if self.args.prob_upto > 0:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[ordered_orig_sorted_category[top_n]], torch.tensor(self.args.prob_upto, dtype=torch.float32))
                            else:
                                c1 = self.args.mse_coeff * mse_loss_fn(outputs[ordered_orig_sorted_category[top_n]], kwargs['target'][ordered_orig_sorted_category[top_n]])
                    else:
                        c1 = self.args.mse_coeff * mse_loss_fn(outputs, kwargs['target'][category].type(torch.LongTensor).unsqueeze(0),)

                c2 = self.args.l1_coeff * torch.mean(torch.abs(mask)) * float(self.args.enable_budget)
                c3 = self.args.tv_coeff * tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)
                mse_var = np.var(metrics['MSE']) if len(metrics['MSE']) > 1 else 0.0
                # -0.8452
                # c4 = self.args.dtw_coeff * softdtw_loss_fn(x=perturbated_input.reshape([1, -1]),y=Zt.reshape([1, -1])) * float(self.args.enable_dtw)
                dist_loss = torch.tensor(0.0)
                if self.args.enable_dist:

                    if self.args.first_class == "yes":
                        dist_data_class = orig_first_category
                    elif self.args.second_class == "yes":
                        dist_data_class = orig_second_category
                    elif self.args.third_class == "yes":
                        dist_data_class = orig_third_category
                    else:
                        dist_data_class = ordered_orig_sorted_category[top_n]

                    dist_class_indices = getattr(self.args.dataset, f"test_class_{dist_data_class}_indices")
                    dist_class_index = np.random.randint(0, len(dist_class_indices))
                    dist_class_data = getattr(self.args.dataset, f"test_class_{dist_data_class}_data")
                    dist_data = torch.tensor(dist_class_data[dist_class_index].reshape(1, -1), dtype=torch.float32).to('cpu')

                    if self.args.dist_loss == 'euc':
                        dist_loss = self.args.dist_coeff * \
                            mse_loss_fn(dist_data.reshape(
                                [1, -1]), perturbated_input.reshape([1, -1]))
                    elif self.args.dist_loss == 'w_euc':
                        dist_loss = self.args.dist_coeff * self.weighted_mse_loss(input=dist_data.reshape(
                            [1, -1]), target=perturbated_input.reshape([1, -1]), weight=upsampled_mask.reshape([1, -1]))
                    elif self.args.dist_loss == 'dtw':
                        dist_loss = self.args.dist_coeff * \
                            softdtw_loss_fn(x=perturbated_input.reshape(
                                [1, -1]), y=dist_data.reshape([1, -1]))
                    elif self.args.dist_loss == 'w_dtw':
                        dist_loss = self.args.dist_coeff * softdtw_loss_fn(x=perturbated_input.reshape(
                            [1, -1]), y=dist_data.reshape([1, -1]), w=upsampled_mask.reshape([1, -1]))
                    elif self.args.dist_loss == 'n_dtw':
                        dist_loss = self.args.dist_coeff * softdtw_loss_fn(x=perturbated_input.reshape(
                            [1, -1]), y=dist_data.reshape([1, -1]), normalize=True)
                    elif self.args.dist_loss == 'n_w_dtw':
                        dist_loss = self.args.dist_coeff * softdtw_loss_fn(x=perturbated_input.reshape(
                            [1, -1]), y=dist_data.reshape([1, -1]), normalize=True, w=upsampled_mask.reshape([1, -1]))
                # if( math.isinf(c1)):
                #     break

                loss = (c1 + c2 + c3 + dist_loss)
                if loss < 0:
                    rs_prios = loss * (rs_weight[0])*0.0
                    ro_prios = loss * (ro_weight[0])*0.0
                else:
                    rs_prios = loss * (rs_weight[0])
                    ro_prios = loss * (ro_weight[0])

                loss = loss * (rs_weight[0] + ro_weight[0]) / 2

                abs_mask_t = mask.detach().clone().numpy()
                mask_min_t= np.min(abs_mask_t)
                mask_max_t = np.max(abs_mask_t)

                delta_t = mask_max_t-mask_min_t
                if (delta_t) == 0:
                    add_noise = 1e-27
                else:
                    add_noise = 0
                norm_mask_t = (abs_mask_t - mask_min_t) / (mask_max_t - mask_min_t + add_noise)
                # old_perturbated_input = perturbated_input.detach().clone().numpy()

                optimizer.zero_grad()
                loss.backward()
                metrics['Mean Gradient'].append(float(np.mean(mask.grad.cpu().detach().numpy())))
                metrics['TV Norm'].append(float(c3.item()))
                metrics['MSE'].append(float(c1.item()))
                metrics['DIST'].append(float(dist_loss.item()))
                metrics['Budget'].append(float(c2.item()))
                # metrics['DTW'].append(float(c4.item()))
                metrics['Total Loss'].append(float(loss.item()))
                metrics['Category'].append(int(np.argmax(outputs.cpu().detach().numpy())))
                metrics['Saliency Sum'].append(float(np.sum(mask.cpu().detach().numpy())))
                metrics['Saliency Var'].append(float(np.var(mask.cpu().detach().numpy())))
                metrics['MSE Coeff'].append(self.args.mse_coeff)
                metrics['TV Coeff'].append(self.args.tv_coeff)
                metrics['L1 Coeff'].append(self.args.l1_coeff)
                # metrics['DTW Coeff'].append(self.args.dtw_coeff)
                metrics['MSE Var'].append(mse_var)

                torch.set_printoptions(precision=2)

                if self.args.top_n == 'yes':
                    reached_prob = outputs[ordered_orig_sorted_category[top_n]]
                    reached_prob_category = ordered_orig_sorted_category[top_n]
                    reached_prob_upto = kwargs['target'].cpu().data.numpy()[ordered_orig_sorted_category[top_n]]
                    multiplier_prob = reached_prob_upto
                else:

                    if self.args.first_class == 'yes':
                        reached_prob = outputs[orig_first_category]
                        reached_prob_category = orig_first_category
                        reached_prob_upto = first_category_pred_prob
                        multiplier_prob = first_category_pred_prob

                    elif self.args.second_class == 'yes' :
                        reached_prob = outputs[orig_second_category]
                        reached_prob_category = orig_second_category
                        reached_prob_upto = second_category_pred_prob
                        multiplier_prob = second_category_pred_prob
                    elif self.args.third_class == 'yes' :
                        reached_prob = outputs[orig_third_category]
                        reached_prob_category = orig_third_category
                        reached_prob_upto = third_category_pred_prob
                        multiplier_prob = third_category_pred_prob
                    else:
                        pass

                
                if self.args.prob_upto >0:
                    reached_prob_upto = self.args.prob_upto

                if self.args.print == 'yes':
                    print(
                        f"Iter: {i}/{self.args.max_itr} | ({i / self.args.max_itr * 100:.2f}%)"
                        # f"| LR: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}"
                        f"| MSE: {metrics['MSE'][-1]:.4f}, V:{mse_var:.4f}"
                        # f"| {self.args.dist_loss}: {metrics['DIST'][-1]:.4f}"
                        f"| top_n:{top_n}"
                        # f"| B: {metrics['Budget'][-1]:.4f} | TL: {metrics['Total Loss'][-1]:.4f} "
                        # f"| C: {metrics['Category'][-1]}"
                        # f"| Confidence: {metrics['Confidence'][-1]:.2f}"
                        # f"| S:{metrics['Saliency Sum'][-1]:.4f}"
                        # f"| RSI: {rs_index[0]} WSI: {rs_weight[0]:.4f}",
                        # f"| ROI: {ro_index[0]} WOI: {ro_weight[0]:.4f}"
                        # f"| EPS: {self.eps:.2f} | M: {self.mode}",
                        f"| Label : {ordered_orig_sorted_category[top_n]} Reached Prob:{outputs[ordered_orig_sorted_category[top_n]]:.4f}"
                        f"| Should reach: {kwargs['target'].cpu().data.numpy()[ordered_orig_sorted_category[top_n]]:.4f}"
                        f"| Prob_upto:{reached_prob_upto:.4f}"
                        # f"| I : {orig_first_category} {outputs[orig_first_category]:.4f}| II: {orig_second_category} {outputs[orig_second_category]:.4f}| III:{orig_third_category} {outputs[orig_third_category]:.4f}"
                        f"| Perturb: {perturbated_input.data}"
                        f"| Norm Mask : {norm_mask_t[0][0]}"
                        # f"| Mask : {mask.data}"
                        # f"| abs Mask : {abs_mask_t}"
                        # f"| Reached:{reached_prob:.4f}"
                        # end="\r",
                        # flush=True
                        # f"| DTW: {c4.item():.2f}"
                        # f"\r"
                    )

                optimizer.step()

                #if ro_index[0] > len(self.ro_priority_buffer.priorities):
                #    print(self.ro_priority_buffer.priorities)
                #    print(len(self.ro_priority_buffer.priorities))
                #    print(ro_index)
                #    print(len(ro_index))
                #    print(f"ro_index {ro_index} len {len(self.ro_priority_buffer.priorities)}")

                self.rs_priority_buffer.update_priorities(rs_index, [rs_prios.item()])
                self.ro_priority_buffer.update_priorities(ro_index, [ro_prios.item()])

                if self.args.enable_lr_decay:
                    scheduler.step(epoch=i)

                # Optional: clamping seems to give better results
                # if self.args.mask_norm == 'clamp':
                #     mask.data.clamp_(0, 1)
                # elif self.args.mask_norm == 'sigmoid':
                #     mask.data = torch.nn.Sigmoid()(mask)
                # elif self.args.mask_norm == 'softmax':
                #     mask.data = torch.nn.Softmax(dim=-1)(mask)
                # elif self.args.mask_norm == 'none':
                #     pass
                mask.data.clamp_(-1, 1)
                # mask.data.clamp_(0, 1)

                if self.args.run_eval_every_epoch:
                    m = mask.cpu().detach().numpy().flatten()
                    metrics["epoch"][f"epoch_{i}"] = {'eval_metrics': run_evaluation_metrics(self.args.eval_replacement, kwargs['dataset'], data, self.predict_fn, m, kwargs['save_dir'], False)}

                    if kwargs['save_perturbations']:
                        self.perturbation_manager.add_perturbation(
                            perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                            saliency=mask.cpu().detach().numpy(),
                            step=i, confidence=metrics['Confidence'][-1],
                            insertion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Insertion']['trap'],
                            deletion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Deletion']['trap'],
                            final_auc=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Final']['AUC'],
                            mean_gradient=metrics['Mean Gradient'][-1],
                            tv_norm=metrics['TV Norm'][-1],
                            main_loss=metrics["MSE"][-1],
                            budget=metrics["Budget"][-1],
                            total_loss=metrics["Total Loss"][-1],
                            category=metrics["Category"][-1],
                            saliency_sum=metrics["Saliency Sum"][-1],
                            saliency_var=metrics["Saliency Var"][-1])

                if self.enable_wandb:
                    _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"},
                            **{"Gradient": wandb.Histogram(mask.grad.cpu().detach().numpy()),
                                "Training": [upsampled_mask, noise, perturbated_input,
                                            ] + [perturbation] if self.args.enable_blur else []
                                }
                            }
                    if f"epoch_{i}" in metrics["epoch"]:
                        _mets = {**_mets, **metrics["epoch"]
                                [f"epoch_{i}"]['eval_metrics']}

                    wandb.log(_mets)

                if self.args.tsne == 'yes':
                    sorted_category = np.argsort(outputs.cpu().data.numpy())
                    reached_prob_category = sorted_category[-1]
                    reached_prob = outputs[reached_prob_category]

                CRED = '\033[91m'
                CEND = '\033[0m'
                CGRN = '\033[92m'
                CGRN = '\033[92m'
                CCYN = "\033[36m"
                #TODO check mse loss and also DTW loss, both should be less
                #TODO first, mask_init should start with second class or third class between 0 to 1 or normalize it
                #TODO Second, check DTW loss also for early stop and not just loss or mse
                #TODO Third, GAN's usage, currently I use MSE as descriminator
                #TODO Important learning if we collect 10% or 500 reached prob then average or mean looks good else it is not valid

                # https://pkg.go.dev/github.com/whitedevops/colors

                # we can use early stopping creteria
                #check DTW loss too and not just mse loss and reached_prob close to prob_upto
            # label_category_pred_prob = kwargs['target'].cpu().data.numpy()[label]
            # first_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_first_category]
            # second_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_second_category]
            # third_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_third_category]
            # fourth_category_pred_prob = kwargs['target'].cpu().data.numpy()[orig_fourth_category]

                if self.args.early_stopping:
                    if i > self.args.early_stop_min_epochs+(reached_perturbs*self.args.early_stop_diff):
                        # if (self.args.tsne == 'yes' and ((reached_prob >= (self.args.prob_upto-self.args.early_stop_prob_range)) and (reached_prob <= (self.args.prob_upto+self.args.early_stop_prob_range)))):
                        # if (reached_prob >= reached_prob_upto):
                        # outputs_t = self.softmax_fn(self.predict_fn(data*mask))[0][0]
                        # reached_prob = outputs_t[ordered_orig_sorted_category[top_n]]

                        if self.args.bbm == 'rnn':
                            if self.args.bbmg == 'yes':
                                timesteps = int(perturbated_input.shape[0]/self.args.window_size)
                                outputs_orig = self.softmax_fn(self.predict_fn((data*norm_mask_t).reshape(-1,timesteps,self.args.window_size))).unsqueeze(0)
                            else:
                                outputs_orig = self.softmax_fn(self.predict_fn((data*norm_mask_t).reshape(1,-1))).unsqueeze(0)
                        else:
                            outputs_orig = self.softmax_fn(self.predict_fn(data*norm_mask_t))
                        reached_prob_orig = outputs_orig[0][0][ordered_orig_sorted_category[top_n]]
                        reached_prob_category_orig = ordered_orig_sorted_category[top_n]

                        # if (((reached_prob_orig >= (reached_prob_upto-self.args.early_stop_prob_range)) and (reached_prob <= (reached_prob_upto+self.args.early_stop_prob_range)))):
                        if (((reached_prob_orig >= (reached_prob_upto)))):
                            # print(CRED+f"data*saliency{(data*abs_mask_t).tolist()} Normmask{norm_mask_t.tolist()}"+CEND)
                            # print(CCYN+f"Step: {i} | New Prob {torch.argmax(outputs_orig)} | Num Perturbs: {self.args.num_perturbs} | Prob_upto : {self.args.prob_upto:.4f} | Reached Prob: {reached_prob_orig:.4f}| Should Reach:{reached_prob_upto:.4f} | Prob Range:{self.args.early_stop_prob_range}| Label : {ordered_orig_sorted_category[top_n]} |Pinput {perturbated_input.data}"+CEND)
                            if reached_perturbs <= self.args.num_perturbs:
                                flag = 0
                                reached_perturbs = reached_perturbs + 1
                                various_perturbations.append(perturbated_input.data.cpu().detach().numpy())
                                # various_perturbations.append(old_perturbated_input)
                                # mask_t = mask.detach().clone()
                                mask_t = abs_mask_t 
                                various_perturbations_masks.append(mask_t)
                                various_perturbations_labels.append(reached_prob_category)
                                # plt.xlabel(str(reached_perturbs)+'X_LABEL')
                                # plt.plot(various_perturbations[-1], label="Various Signal ")
                                # # plt.plot(various_perturbations_masks[-1][0][0], label="Various Signal Masks")
                                # plt.show()
                                # plt.savefig(str(reached_perturbs)+'plot.png')

                                # print(CRED+f"various_perturbations:{various_perturbations}"+CEND)
                                # print(CRED+f"various_perturbations_mask:{various_perturbations_masks}"+CEND)
                            else:
                                continue
                                if(self.args.old_auc == 'yes'):
                                    continue
                                if self.args.new_auc == 'yes':
                                    continue
                                else:
                                    break
                                #TODO until 5000 iterations take all samples which crosses >0.99 for generative samples why only first 10
                                # flag = 1
                                # various_perturbations_mean = np.mean(various_perturbations,axis=0)
                                # various_perturbations_masks_mean = np.mean(various_perturbations_masks,axis=0)
                                # self.save_perturbations_original_timeseries(
                                #     kwargs=kwargs, various_perturbations=various_perturbations, various_perturbations_masks=various_perturbations_masks,
                                #     orig_first_category=orig_first_category, orig_second_category=orig_second_category, orig_third_category=orig_third_category,reached_perturbs=reached_perturbs,
                                #     outputs=outputs)
                                # np.set_printoptions(suppress=True)
                                # np.set_printoptions(precision=6)
                                # print(CRED+f"various_perturbations:{np.array(various_perturbations)}"+CEND)
                                # print(CGRN+f"various_perturbations_mean:{various_perturbations_mean}"+CEND)
                                # print(CRED+f"various_perturbations_masks:{np.array(various_perturbations_masks)}"+CEND)
                                # print(CGRN+f"various_perturbations_masks_mean:{various_perturbations_masks_mean}"+CEND)
                                # various_perturbs_pred = self.softmax_fn(self.predict_fn(torch.tensor(various_perturbations_mean,dtype=torch.float32).reshape([1, -1])))
                                # print(CRED+f"Various Perturbations Mean Predictions: {various_perturbs_pred}")
                                # print(f"outputs| I : {orig_first_category} {outputs[orig_first_category]:.4f}| II: {orig_second_category} {outputs[orig_second_category]:.4f}| III:{orig_third_category} {outputs[orig_third_category]:.4f}")
                                # print(f"various| I : {orig_first_category} {various_perturbs_pred[0][orig_first_category]:.4f}| II: {orig_second_category} {various_perturbs_pred[0][orig_second_category]::.4f.4f}| III:{orig_third_category} {various_perturbs_pred[0][orig_third_category]:.4f}")
                                # break

            # mask_t = various_perturbations[-1]

            # print(f'various perturbations masks {various_perturbations_masks}')

            if(len(various_perturbations) ==0):
                various_perturbations_mean = np.zeros(len(data))
                various_perturbations_masks_mean = np.zeros([1,1,len(data)])
                various_perturbations_masks_mean = mask.data.numpy()
                metrics['var_pert_len'].append(float(1))
            else:
                metrics['var_pert_len'].append(float(len(various_perturbations)))
                various_perturbations_mean = np.mean(various_perturbations, axis=0)
                if(len(various_perturbations) >self.args.sample_masks):
                    if self.args.clusters == 'mean':
                        various_perturbations_masks_mean = np.mean(various_perturbations_masks[-self.args.sample_masks:], axis=0)
                    else:
                        km = np.array(various_perturbations_masks[-self.args.sample_masks:])
                        km = km.squeeze()
                        if km.size == 0 or km[0].size == 0:
                            various_perturbations_masks_mean = np.mean(various_perturbations_masks[-self.args.sample_masks:], axis=0)
                        else:
                            km = km.squeeze()
                            ts = len(km[0])
                            km = km.reshape(-1,ts,1)
                            if self.args.cluster_dtw  == 'dtw':
                                km_dba = TimeSeriesKMeans(n_clusters=self.args.n_clusters, metric="dtw", max_iter=5,max_iter_barycenter=5,random_state=0).fit(km)
                            else:
                                km_dba = TimeSeriesKMeans(n_clusters=self.args.n_clusters, metric="softdtw", max_iter=5,max_iter_barycenter=5,metric_params={"gamma": .5},random_state=0).fit(km)
                            if(self.args.cluster_indx >= self.args.n_clusters):
                                various_perturbations_masks_mean = km_dba.cluster_centers_[0].reshape(1,-1).reshape(1,1,ts)
                            else:
                                various_perturbations_masks_mean = km_dba.cluster_centers_[self.args.cluster_indx].reshape(1,-1).reshape(1,1,ts)

                else:
                    various_perturbations_masks_mean = mask.data.numpy()
            # various_perturbations_masks_mean = various_perturbations_masks[0]
            # various_perturbations_mean = various_perturbations[-1]
            # various_perturbations_masks_mean = various_perturbations_masks[-1]
            # print(f'various perturbations masks mean{various_perturbations_masks_mean}')
            # various_perturbations_masks_mean = np.mean(various_perturbations_masks, axis=0)
            if self.args.old_auc == 'yes':
                various_perturbations_masks_mean = mask.data.numpy()
            if self.args.new_auc == 'yes':
                various_perturbations_masks_mean = mask.data.numpy()

            mask_t = various_perturbations_masks_mean
            # mask_t = various_perturbations_masks
            # print(f'top_n{top_n} various perturbation {mask_t} category{ordered_orig_sorted_category[top_n]}')
            #instead of mean use centroid 
            top_n_masks.append(mask_t)
            top_n_various_perturbations.append(various_perturbations_mean)
            abs_mask_t = mask_t 
            mask_min_t= np.min(abs_mask_t)
            mask_max_t = np.max(abs_mask_t)

            delta_t = mask_max_t-mask_min_t
            if (delta_t) == 0:
                add_noise = 1e-27
            else:
                add_noise = 0
            norm_mask_t = (mask_t - mask_min_t) / (mask_max_t - mask_min_t + add_noise)
            # print(f'norm_mask_t various perturbs {norm_mask_t}')
            if delta_t == 0:
                top_n_norm_masks.append(np.zeros([1,1,len(data)]))
            else:
                top_n_norm_masks.append(norm_mask_t)

            if self.args.nth_prob == 0.0:
                top_nth_prob =  kwargs['target'].cpu().data.numpy()[ordered_orig_sorted_category[top_n]]
            else:
                top_nth_prob = self.args.nth_prob
               
            if top_n >0:
                top_n_prob_norm_masks.append(top_nth_prob*norm_mask_t)
                # top_n_prob_norm_masks.append(1*norm_mask_t)
            else:
                top_n_prob_norm_masks.append(norm_mask_t)

        print(f'\n\tData {data.data}') 
        print(f'\n\tOrdered category{ordered_orig_sorted_category}')
        print(f'\n\tOriginal Pred {orig_pred_prob.tolist()}')
        # print(f'normalized mask* multiplier_prob {norm_mask.flatten()*multiplier_prob}')
        print("\n\t Top N Masks")
        #for i in range(len(top_n_masks)):
            #print([f'{(x):.4f}' for x in top_n_masks[i][0][0]])
        # print(f'\n\ttop n masks{np.array2string(np.array(top_n_masks))}')
        print("\n\t Top N Norm Masks")
        #for i in range(len(top_n_norm_masks)):
            #print([f'{(x):.4f}' for x in top_n_norm_masks[i][0][0]])
        # print(f'\n\ttop n norm masks{np.array2string(np.array(top_n_norm_masks))}')
        # print(f'\n\ttop n prob norm masks{np.array2string(np.array(top_n_prob_norm_masks))}')
        print("\n\t Top N Prob Norm Masks")
        #for i in range(len(top_n_prob_norm_masks)):
            #print([f'{(x):.4f}' for x in top_n_prob_norm_masks[i][0][0]])

        print("\n\t Top N various Mean perturbations")
        #for i in range(len(top_n_various_perturbations)):
            #print([f'{(x):.4f}' for x in top_n_various_perturbations[i]])

        print(f'\n\tTop_n: Main Saliency Generation') 
        for top_n in range(top_n_range):

            # if(top_n >=1):
            #     continue

            if(np.isnan(top_n_prob_norm_masks[top_n]).any()):
                continue

            if(top_n > 0):
                if self.args.preserve_prob == 'yes':
                    main_saliency_t = main_saliency-top_n_prob_norm_masks[top_n]
                    main_saliency_t = main_saliency_t.astype(np.float32)
                    if self.args.bbm == 'rnn':
                        # new_prob_pred = self.softmax_fn(self.predict_fn(data*main_saliency_t))
                        new_prob_pred = self.softmax_fn(self.predict_fn((data*main_saliency).reshape(1,-1))).unsqueeze(0)
                    else:
                        new_prob_pred = self.softmax_fn(self.predict_fn(data*main_saliency_t))
                    old_first_pred_prob = kwargs['target'].cpu().data.numpy()[orig_first_category]
                    new_first_pred_prob = new_prob_pred[0][0][orig_first_category]
                    if(new_first_pred_prob < old_first_pred_prob):
                        break
                else:
                    main_saliency = main_saliency - top_n_prob_norm_masks[top_n]
                # pass
            else:
                main_saliency = top_n_prob_norm_masks[top_n]
            #print([f'{(x):.2f}' for x in main_saliency[0][0]])
        

            # print(f'Top_n_prob_norm_mask:{top_n_prob_norm_masks[top_n].tolist()}')

            # print(f'\n \t Outputs Pred top_n:_{top_n}') 
            # print(f'{self.softmax_fn(self.predict_fn(data*main_saliency)).tolist()}')

        # data = torch.tensor([1,0,0,0,1,0,0,0,0,0],dtype=torch.float32)
        # data= torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]],dtype=torch.float32)

        if self.args.old_auc == 'yes':
            main_saliency = top_n_prob_norm_masks[0]
        # if self.args.new_auc == 'yes':
        #     main_saliency = top_n_prob_norm_masks[0]

        #print(f'\n \t Data {data.data}')
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        #print(f'\n \t Final Main unique saliency')
        #print([f'{(x):.2f}' for x in main_saliency[0][0]])

        # main_saliency = np.clip(main_saliency,0,1)
        main_saliency = main_saliency

        #print(f'\n \t Final Main unique Norm saliency')
        #print([f'{(x):.2f}' for x in main_saliency[0][0]])


        #print(f'\n \t Final Main saliency*Data')
        #print([f'{(x):.2f}' for x in data*main_saliency[0][0]])

        main_saliency = main_saliency.astype(np.float32)
        if self.args.bbm == 'rnn':
            if self.args.bbmg == 'yes':
                timesteps = int(perturbated_input.shape[0]/self.args.window_size)
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*main_saliency).reshape(-1,timesteps,self.args.window_size))).unsqueeze(0)
            else:
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*main_saliency).reshape(1,-1))).unsqueeze(0)
        else:
            Final_unique_prob_pred = self.softmax_fn(self.predict_fn(data*main_saliency))
        #print(f'\n \t Final Unique Outputs Pred') 
        #print([f'{(x):.4f}' for x in Final_unique_prob_pred[0][0]])
        # print(f'\n \t Original data Outputs Pred ')
        # print(f'{self.softmax_fn(self.predict_fn(data)).tolist()}')
        #print(f'\n \t Original Pred kwargs outputs')
        #print(orig_pred_prob)
        # various_perturbs_pred = self.softmax_fn(self.predict_fn(torch.tensor(various_perturbations_mean,dtype=torch.float32).reshape([1, -1])))
        # outputs = (self.predict_fn(perturbated_input.reshape([1, -1])))
        #print(CRED+f'\n \t Difference in pred prob'+CEND) 
        Final_diff = Final_unique_prob_pred - orig_pred_prob
        #print(CRED+f'{Final_diff}'+CEND)

        #print(f'Reached_perturbs {reached_perturbs} various_len {len(various_perturbations)} sample_masks:{self.args.sample_masks}')
        delta_pred_prob = Final_unique_prob_pred[0][0][orig_first_category] - kwargs['target'].cpu().data.numpy()[orig_first_category]

        upsampled_mask = torch.tensor(main_saliency,dtype=torch.float32)
        mask = torch.tensor(main_saliency,dtype=torch.float32)

        np.save(kwargs['save_dir'] + "/upsampled_mask", upsampled_mask.cpu().detach().numpy())
        np.save(kwargs['save_dir'] + "/mask", mask.cpu().detach().numpy())
        if self.enable_wandb:
            wandb.save(kwargs['save_dir'] + "/metrics.json")
            wandb.save(kwargs['save_dir'] + "/upsampled_mask.npy")
            wandb.save(kwargs['save_dir'] + "/mask.npy")

        mask = mask.squeeze(0).squeeze(0)
        mask = mask.cpu().detach().numpy().flatten()

        if self.enable_wandb:
            wandb.run.summary["prediction_probability_diff"] = delta_pred_prob
            wandb.run.summary["pos_features"] = len(np.where(mask > 0)[0])
            wandb.run.summary["neg_features"] = len(np.where(mask < 0)[0])
            wandb.run.summary["pos_sum"] = np.sum(mask[np.argwhere(mask > 0)])
            wandb.run.summary["neg_sum"] = np.sum(mask[np.argwhere(mask < 0)])
        abs_mask = mask  # np.abs(mask)
        mask_min = np.min(abs_mask)
        mask_max = np.max(abs_mask)
        norm_mask = (mask - mask_min) / (mask_max - mask_min)
        #print(f'normalized mask {norm_mask.flatten()}')
        #print(f'mask {mask.flatten()}')
        # mask = norm_mask
        if self.args.bbm == 'rnn':
            if self.args.bbmg == 'yes':
                timesteps = int(perturbated_input.shape[0]/self.args.window_size)
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*norm_mask).reshape(-1,timesteps,self.args.window_size))).unsqueeze(0)
            else:
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*norm_mask).reshape(1,-1))).unsqueeze(0)
        else:
            Final_unique_prob_pred = self.softmax_fn(self.predict_fn(data*norm_mask))

        # print(f'2:data*Norm mask {data*norm_mask}')
        # print(f'2:Final Unique Outputs Pred {self.softmax_fn(self.predict_fn(data*norm_mask))}')
        # print(f'Difference in pred prob {(Final_unique_prob_pred - orig_pred_prob)}')
        # print(f'data {data}')
        # print(f"Probs {(kwargs['target'].cpu().data.numpy())}")
        save_timeseries(mask=norm_mask, raw_mask=mask, time_series=data.numpy(),blurred=blurred_signal, save_dir=kwargs['save_dir'], enable_wandb=self.enable_wandb,algo=self.args.algo, dataset=self.args.dataset, category=label)

        # if self.args.early_stopping and flag == 0:
        #     self.save_perturbations_original_timeseries(
        #         kwargs=kwargs, various_perturbations=various_perturbations, various_perturbations_labels = various_perturbations_labels, various_perturbations_masks=various_perturbations_masks,
        #         orig_first_category=orig_first_category, orig_second_category=orig_second_category, orig_third_category=orig_third_category,reached_perturbs=reached_perturbs,
        #         outputs=outputs)

        # Not a good idea
        # p = float(outputs[category].item())
        # mask_indices = {i: e for e, i in enumerate(np.argsort(mask)[::-1])}
        # sorted_mask = np.sort(mask)[::-1]
        # sorted_mask = sorted_mask[:math.ceil(len(mask) * p)]
        # sorted_mask = np.array([*sorted_mask, *np.zeros(len(mask) - len(sorted_mask))])
        # sorted_reverted_mask = list(range(len(mask)))
        # for i,e in mask_indices.items():
        #     sorted_reverted_mask[i]=sorted_mask[e]
        # mask = np.array(sorted_reverted_mask)
        # print(np.argsort(mask)[::-1])

        if self.enable_wandb:
            wandb.run.summary["norm_saliency_sum"] = np.sum(mask)
            wandb.run.summary["norm_saliency_var"] = np.var(mask)
            wandb.run.summary["norm_pos_sum"] = np.sum(norm_mask[np.argwhere(mask > 0)])
            wandb.run.summary["norm_neg_sum"] = np.sum(norm_mask[np.argwhere(mask < 0)])
            wandb.run.summary["label"] = label

        np.save("./r_mask", mask)
        np.save("./n_mask", norm_mask)

        # orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy())
        return mask, self.perturbation_manager