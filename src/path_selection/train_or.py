import torch.optim as optim
from net import *
from loss import *
from sklearn.svm import SVC
import argparse
import numpy as np
from utils import *
from configparser import ConfigParser
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, help="the config file", default="./config/train.config"
)
args = parser.parse_args()


def train_AE_withC(net, feature, p_q, p_Rq, config):
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=config.getfloat("trainsetting", "lr"))
    objective_func = AEloss()

    for i in range(1, config.getint("trainsetting", "epoch") + 1):

        optimizer.zero_grad()
        output = net(feature, c_state=True, c_cluster_state=True)

        loss = objective_func(
            feature,
            output["encode"],
            output["decode"],
            output["C1"],
            output["C2"],
            output["latent_c"],
            output["latent_cluster"],
            output["cluster_center"],
        )

        total_loss = (
            loss["recon_loss"]
            + 10 * loss["diag_C1_loss"]
            + p_q * (loss["C1_loss"] + loss["C2_loss"])
            + p_Rq * (loss["self_C1_loss"] + loss["self_C2_loss"])
        )

        total_loss.backward()
        optimizer.step()

        if (i % 10) == 0:
            print(
                "%d T:%.4f R:%.4f Q:%.4f RQ:%.4f P:%.4f RP:%.4f DQ:%.4f "
                % (
                    i,
                    total_loss.item(),
                    loss["recon_loss"],
                    loss["C1_loss"],
                    loss["C2_loss"],
                    loss["self_C1_loss"],
                    loss["self_C2_loss"],
                    loss["diag_C1_loss"],
                )
            )
    return net


def ttest_AE_withC(net, train_data, test_data, train_label, test_label, f):
    net.eval()
    parameter = net.state_dict()
    C1 = parameter["coeff"].cpu().detach().numpy()
    C2 = parameter["coeff_cluster"].cpu().detach().numpy()

    C1 = C1 * C1
    C2 = C2 * C2
    s1 = np.sum(C1, 0)
    s2 = np.sum(C2, 0)

    s1 = (s1 - np.min(s1)) / (np.max(s1) - np.min(s1))
    s2 = (s2 - np.min(s2)) / (np.max(s2) - np.min(s2))

    s_ours = np.argsort(-(s1 + s2))

    clr = SVC(C=100, kernel="linear")

    acc_svm = []
    query = np.array(range(5, 80 + 1, 5), dtype=int)
    for i in query:
        clr.fit(train_data[s_ours[:i]], train_label[s_ours[:i]])
        acc_svm.append(clr.score(test_data, test_label))

    result = ""
    for item in acc_svm:
        result += str(item)[:5] + ", "
    result += str(np.sum(acc_svm))[:5]
    result += "\n"

    f.write(result)

    return net


if __name__ == "__main__":

    config = ConfigParser()
    config.read(args.config)

    # Load Data
    print("Load Data....")
    train_data = np.load(config["path"]["data_path"] + "1fea1.npy")
    train_label = np.load(config["path"]["data_path"] + "1lab1.npy")
    test_data = np.load(config["path"]["data_path"] + "1fea2.npy")
    test_label = np.load(config["path"]["data_path"] + "1lab2.npy")

    feature = torch.Tensor(train_data)
    feature_ = torch.Tensor(test_data)

    if torch.cuda.is_available():
        feature = feature.cuda()
        feature_ = feature_.cuda()

    batch_size = len(feature)
    print("Done!")

    p_q = json.loads(config.get("trainsetting", "p_q"))
    p_Rq = json.loads(config.get("trainsetting", "p_Rq"))

    print("Start trainging!")
    for i in p_q:
        for j in p_Rq:
            f = open(config["path"]["log_save_path"] + "results.txt", "a")
            parameter = "Q:" + str(i) + "  " + "RQ:" + str(j) + "\n"
            # f.write(parameter)

            print("Inti Net....")
            net = AutoEncoder(
                batch_size,
                config.getint("trainsetting", "k"),
                json.loads(config.get("trainsetting", "layers")),
            )
            pre_dict = torch.load(config["path"]["model_load_path"] + "only_AE.pt")
            model_dict = net.state_dict()
            pre_dict = {
                k: v
                for k, v in pre_dict.items()
                if (k in model_dict) and (k not in ["coeff", "coeff_cluster", "kmeans"])
            }
            model_dict.update(pre_dict)
            net.load_state_dict(model_dict)

            if torch.cuda.is_available():
                net = net.cuda()
            print("Done!")

            net.eval()
            train_data = net.encoder(feature).cpu().detach().numpy()
            test_data = net.encoder(feature_).cpu().detach().numpy()

            net = train_AE_withC(net, feature, i, j, config)
            net = ttest_AE_withC(net, train_data, test_data, train_label, test_label, f)
            f.close()
