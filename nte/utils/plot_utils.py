import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
from nte.metrics.saliency_frequency import salient_frequency

SNS_CMAP = ListedColormap(sns.light_palette('red').as_hex())
import matplotlib.pyplot as PLT


def intialize_plot():
    try:
        PLT.style.use("presentation")
    except Exception:
        pass
    PLT.gcf().clf()
    PLT.figure(figsize=(15, 6))
    return PLT


def visualize_all_classes(data, label, display=True, title=''):
    data, indices = np.unique(data, axis=0, return_index=True)
    label = label[indices]
    
    plt = intialize_plot()
    total_candidates = 0
    zero_indices = np.argwhere(label==0)
    one_indices = np.argwhere(label==1)
    fig = plt.figure(figsize=(15, 2*max(len(zero_indices), len(one_indices))))
    for e, i in enumerate(zero_indices):
        plt.subplot(len(data),2, e+1)
        if e==0:
            plt.title('Class 0 Patterns', size=18)
        plt.plot(data[i].flatten())
        total_candidates+=1

    for e, i in enumerate(one_indices):
        plt.subplot(len(data),2, e+total_candidates+1)
        if e==0:
            plt.title('Class 1 Patterns', size=18)
        plt.plot(data[i].flatten(), color='red')
        total_candidates += 1

    fig.suptitle(title, y=1.03, size=18)
    plt.tight_layout()
    if display:
        plt.show()

def plot_saliency_cmap_multi(data, raw_weights, weights, display=True, plt=None, dataset_name="Dataset",
                       labels=['SHAP', 'Lime', 'L-Sal', 'C-Sal', 'GLE']):
    if plt is None:
        plt = PLT
        intialize_plot()
        plt.figure(figsize=(10, 1.5*len(data) if len(data.shape)>1 else 1.5))

    fig = plt.gcf()
    if len(data.shape) > 1:
        timesteps = data.shape[1]
        if len(labels) != data.shape[0]:
            labels = [f"{i}" for i in range(data.shape[0])]
        total_graphs = data.shape[0]
        for e, (d, s) in enumerate(zip(data, weights)):
            plt.subplot(total_graphs, 1, e + 1)
            if e != len(data) - 1:
                plt.xticks(color='w')
            plt.imshow(s[np.newaxis, :], cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                       extent=[0, len(s)-1, float(np.min([np.min(d), np.min(s)]))-1e-1,float(np.max([np.max(d), np.max(s)]))+1e-1])
            plt.plot(d, lw=4)
            plt.grid(False)
            plt.ylabel(labels[e])
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

    else:
        timesteps = data.shape[0]
        plt.imshow(raw_weights[np.newaxis, :], cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                   extent=[0, len(weights)-1, float(np.min([np.min(data), np.min(weights)]))-1e-1, float(np.max([np.max(data), np.max(weights)]))+1e-1])
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        # plt.figure().colorbar(im, cax=cax, orientation='horizontal')
        plt.plot(data, lw=4)
        plt.grid(False)
    if display:
        plt.xlabel("Timesteps")
        plt.grid(False)
        cax = fig.add_axes([0.1, 1.1, 0.5, 0.05])
        plt.text(-0.15, 0, "Saliency", fontsize=22)
        plt.text(-1, 0, "Low")
        plt.text(0.95, 0, "High")
        plt.text(1.5, 0, dataset_name, fontsize=38)
        im = plt.imshow(np.array([i / 100 for i in range(-100, 100, 1)])[np.newaxis, :], cmap=SNS_CMAP, aspect="auto",
                        alpha=0.85,
                        extent=[0, timesteps, 0, 1])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.tight_layout(pad=1)
        plt.show()



def plot_saliency_cmap(data, weights, display=True, plt=None, dataset_name="Dataset",
                       labels=['SHAP', 'Lime', 'L-Sal', 'C-Sal', 'GLE']):
    if plt is None:
        plt = PLT
        intialize_plot()
        plt.figure(figsize=(10, 1.5*len(data) if len(data.shape)>1 else 1.5))

    fig = plt.gcf()
    if len(data.shape) > 1:
        timesteps = data.shape[1]
        if len(labels) != data.shape[0]:
            labels = [f"{i}" for i in range(data.shape[0])]
        total_graphs = data.shape[0]
        for e, (d, s) in enumerate(zip(data, weights)):
            plt.subplot(total_graphs, 1, e + 1)
            if e != len(data) - 1:
                plt.xticks(color='w')
            plt.imshow(s[np.newaxis, :], cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                       extent=[0, len(s)-1, float(np.min([np.min(d), np.min(s)]))-1e-1,float(np.max([np.max(d), np.max(s)]))+1e-1])
            plt.plot(d, lw=4)
            plt.grid(False)
            plt.ylabel(labels[e])
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

    else:
        timesteps = data.shape[0]
        plt.imshow(weights[np.newaxis, :], cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                   extent=[0, len(weights)-1, float(np.min([np.min(data), np.min(weights)]))-1e-1, float(np.max([np.max(data), np.max(weights)]))+1e-1])
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        # plt.figure().colorbar(im, cax=cax, orientation='horizontal')
        plt.plot(data, lw=4)
        plt.grid(False)
    if display:
        plt.xlabel("Timesteps")
        plt.grid(False)
        cax = fig.add_axes([0.1, 1.1, 0.5, 0.05])
        plt.text(-0.15, 0, "Saliency", fontsize=22)
        plt.text(0, 0, "Low")
        plt.text(0.95, 0, "High")
        plt.text(1.5, 0, dataset_name, fontsize=38)
        im = plt.imshow(np.array([i / 100 for i in range(100)])[np.newaxis, :], cmap=SNS_CMAP, aspect="auto",
                        alpha=0.85,
                        extent=[0, timesteps, 0, 1])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.tight_layout(pad=1)
        plt.show()


def plot_salient_frequency_cmap(weights, display=True, plt=None, dataset_name="Dataset",
                                labels=['SHAP', 'Lime', 'L-Sal', 'C-Sal', 'GLE']):
    print("====" * 14)

    def derive_density(saliencies):
        saliencies = np.array([int(i * 100) for i in saliencies])
        bins = [i for i in range(0, 100, 5)]
        dig = np.digitize(saliencies, bins)
        dig_u = np.unique(dig, return_counts=True)
        dig_u_dict = {}
        for k, v in zip(*dig_u):
            dig_u_dict[k] = v
        for nbin in range(len(bins)):
            if nbin not in dig_u_dict.keys():
                dig_u_dict[nbin] = 0
        return {k: dig_u_dict[k] for k in sorted(dig_u_dict)}

    if plt is None:
        plt = PLT
    fig = plt.gcf()

    if len(weights.shape) > 1:

        if len(labels) != weights.shape[0]:
            labels = [f"{i}" for i in range(weights.shape[0])]
        timesteps = weights.shape[1]
        total_graphs = weights.shape[0]
        for e, s in enumerate(weights):
            freq_dict = derive_density(s)
            norm = len(freq_dict.keys())
            s = np.array(list(freq_dict.values()))
            x_axis = np.array(list(freq_dict.keys())) / norm
            plt.subplot(total_graphs, 1, e + 1)
            if e != len(weights) - 1:
                plt.xticks(color='w')
            plt.imshow(s[np.newaxis, :], cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                       extent=[0, len(s) / norm, np.min(s), np.max(s)])
            plt.plot(x_axis, s, lw=4)
            sf, s_var = salient_frequency(weights[e])
            plt.text(x_axis[-1], np.max(s) / 2, f"Var: {s_var:.2f}\nSF : {sf:.2f}", fontsize=18)
            plt.grid(False)
            plt.ylabel(labels[e])
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])

    else:
        freq_dict = derive_density(weights)
        print(freq_dict)
        norm = len(freq_dict.keys())
        weights = np.array(list(freq_dict.values()))
        x_axis = np.array(list(freq_dict.keys())) / norm

        timesteps = weights.shape[0]
        plt.imshow(weights[np.newaxis, :], cmap=SNS_CMAP, aspect="auto", alpha=0.85,
                   extent=[0, len(weights) / norm, np.min(weights), np.max(weights)])
        # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
        # plt.figure().colorbar(im, cax=cax, orientation='horizontal')
        plt.plot(x_axis, weights, lw=4)
        plt.grid(False)
    if display:
        plt.xlabel("% of Timesteps")
        plt.grid(False)
        cax = fig.add_axes([0.1, 1.1, 0.5, 0.05])
        plt.text(-0.15, 0, "Saliency", fontsize=22)
        plt.text(0, 0, "Low")
        plt.text(0.95, 0, "High")
        plt.text(1.5, 0, dataset_name, fontsize=38)
        im = plt.imshow(np.array([i / 100 for i in range(100)])[np.newaxis, :], cmap=SNS_CMAP, aspect="auto",
                        alpha=0.85,
                        extent=[0, timesteps, 0, 1])
        fig.colorbar(im, cax=cax, orientation="horizontal")
        plt.tight_layout(pad=1)
        plt.show()
