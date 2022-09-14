import torch
import numpy as np
import tqdm
from nte.utils import generate_gaussian_noise
from nte.models import GLESaliencyNetwork, GLESaliencyLoss
from nte.models.saliency_model import Saliency

class GLESaliency(Saliency):
    """
    Gaussian Learner Saliency
    """
    def __init__(self, background_data, background_label, predict_fn, config, verbose=True):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.config=config
        self.verbose = verbose

    def generate_saliency(self, data, label, **kwargs):
        saliency_values = []
        for X in tqdm.tqdm(data):
            noise = torch.tensor(generate_gaussian_noise(X), dtype=torch.float32)
            X = torch.tensor(X, dtype=torch.float32)
            saliency_network = GLESaliencyNetwork(self.config["timesteps"])
            saliency_loss = GLESaliencyLoss(self.config["timesteps"])
            optimizer = torch.optim.Adam(saliency_network.parameters(), lr=self.config["learning_rate"])
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            loss_list, var_list, w_list = [], [], []
            alpha, beta, gamma = 1, 1, 0
            y_star = self.predict_fn(X)
            for epoch in range(self.config["num_epochs"]):
                X = X.reshape([1, self.config["timesteps"]])
                saliency_value = saliency_network(X).reshape([1, self.config["timesteps"]])
                y_hat = self.predict_fn(X + (noise * saliency_value))
                optimizer.zero_grad()
                loss, lm, lb, lv = saliency_loss(y_hat , y_star, saliency_value, X, X + (noise * saliency_value),
                                                 alpha=alpha, beta=beta, gamma=gamma)
                loss.backward(retain_graph=True)
                optimizer.step()
                loss_list.append(loss.item())
                w_list.append(np.sum(saliency_value.detach().numpy()))
                var_list.append(np.var(saliency_value.detach().numpy()))
                if self.verbose and (epoch + 1) % self.config["display_step"] == 0:
                    print(
                        'Epoch [{}/{}], | TL: {:.4f}, ML: {:.4f}, BL: {:.4f}, VL: {:.4f}  '
                        '| Var: {:.4f}, W:{:.4f}'.format(epoch + 1,self.config["num_epochs"],loss.item(),lm.item(),lb.item(),
                                                         lv.item(),var_list[-1],w_list[-1]))
            saliency_values.append(1 - saliency_value.detach().numpy().flatten())
        return np.array(saliency_values)

