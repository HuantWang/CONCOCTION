import torch.nn as nn
import torch
from utils import *
from sklearn.cluster import MiniBatchKMeans


class AutoEncoder(nn.Module):
    """
    AutoEncoder with self-expression layer
    """

    def __init__(self, n_sample, k, layers, norm=False):
        super(AutoEncoder, self).__init__()

        self.norm = norm
        self.kmeans = MiniBatchKMeans(n_clusters=k, batch_size=50, random_state=43)
        self.flag = True
        self.cluster_center = 0

        # set_seed(12351)
        set_seed(123)
        self.coeff = nn.init.xavier_normal_(torch.Tensor(n_sample, n_sample))
        self.coeff = nn.Parameter(self.coeff)

        set_seed(351)
        self.coeff_cluster = nn.init.xavier_normal_(torch.Tensor(k, n_sample))
        self.coeff_cluster = nn.Parameter(self.coeff_cluster)

        num_lay = len(layers)

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for i in range(0, num_lay - 1):
            self.encoder.add_module(
                name="encoder{}".format(i), module=nn.Linear(layers[i], layers[i + 1])
            )
            if self.norm:
                self.encoder.add_module(
                    name="bn{}".format(i), module=nn.BatchNorm1d(layers[i + 1])
                )
            if i < num_lay - 2:
                self.encoder.add_module(name="Relu{}".format(i), module=nn.ReLU())

        for i in range(1, num_lay):
            self.decoder.add_module(
                name="decoder{}".format(i - 1),
                module=nn.Linear(layers[-i], layers[-i - 1]),
            )
            if i < num_lay - 1:
                self.decoder.add_module(name="Relu{}".format(i - 1), module=nn.ReLU())
                if self.norm:
                    self.decoder.add_module(
                        name="bn{}".format(i), module=nn.BatchNorm1d(layers[-i - 1])
                    )

    def forward(self, x, c_state=True, c_cluster_state=False):
        """
        If c_state == True,
        this network is computed with self-expression layer
        """
        encode = self.encoder(x)
        if c_state:
            latent_c = self.coeff.mm(encode)
            decode = self.decoder(latent_c)
            if c_cluster_state:
                if self.flag:
                    self.cluster_label = self.kmeans.fit_predict(
                        encode.detach().cpu().numpy()
                    )
                    self.cluster_center = self.kmeans.cluster_centers_
                    self.cluster_center = torch.Tensor(self.cluster_center)
                    self.cluster_label = torch.Tensor(self.cluster_label)
                    self.flag = False
                if torch.cuda.is_available():
                    self.cluster_center = self.cluster_center.cuda()
                latent_cluster = self.coeff_cluster.mm(encode)
                return {
                    "encode": encode,
                    "latent_c": latent_c,
                    "C1": self.coeff,
                    "latent_cluster": latent_cluster,
                    "C2": self.coeff_cluster,
                    "cluster_center": self.cluster_center,
                    "decode": decode,
                    "cluster_label": self.cluster_label,
                }
            else:
                return {
                    "encode": encode,
                    "latent_c": latent_c,
                    "C1": self.coeff,
                    "decode": decode,
                }
        else:
            decode = self.decoder(encode)
            return {"encode": encode, "decode": decode}


if __name__ == "__main__":
    """
    Test Code
    """

    # net1 = AutoEncoder(100, 3, [1000, 256, 128, 64])
    net1 = AutoEncoder(100, 3, [147, 128, 64, 32])
    print(net1)
    o = net1(torch.rand(100, 147), c_state=True, c_cluster_state=True)
    print(o["encode"].shape)
