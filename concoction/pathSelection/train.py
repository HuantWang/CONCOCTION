import torch.optim as optim
from net_dual import *
from loss_dual import *
from sklearn.svm import SVC
import argparse
import numpy as np
from utils import *
from configparser import ConfigParser
import json,os



def train_AE_withC(net, feature, p_q, p_Rq, config):

    net.train()
    optimizer = optim.Adam(net.parameters(), lr = config.getfloat('trainsetting', 'lr'))
    objective_func = AEloss()

    for i in range(1, config.getint('trainsetting', 'epoch')+1):
        
        optimizer.zero_grad()
        output = net(feature, c_state=True, c_cluster_state=True)

        loss = objective_func(feature, output['encode'], output['decode'], output['C1'], \
                              output['C2'], output['latent_c'], output['latent_cluster'], \
                              output['cluster_center'])

        total_loss = loss['recon_loss'] + 10*loss['diag_C1_loss'] +\
                     p_q * (loss['C1_loss'] + loss['C2_loss']) + \
                     p_Rq * (loss['self_C1_loss'] + loss['self_C2_loss'])

        total_loss.backward()
        optimizer.step()

        # if (i % 10) == 0:
            # print('%d T:%.4f R:%.4f Q:%.4f RQ:%.4f P:%.4f RP:%.4f DQ:%.4f ' % \
            #       (i, total_loss.item(), loss['recon_loss'], loss['C1_loss'], \
            #        loss['C2_loss'] , loss['self_C1_loss'], loss['self_C2_loss'], loss['diag_C1_loss']))
    return net


def ttest_AE_withC(net, train_data, k_num):
    
    net.eval()
    parameter = net.state_dict()
    C1 = parameter['coeff'].cpu().detach().numpy()
    C2 = parameter['coeff_cluster'].cpu().detach().numpy()

    C1_l2_norm = np.linalg.norm(C1, ord=2)
    C2_l2_norm = np.linalg.norm(C1, ord=2)

    C1 = C1 * C1
    C2 = C2 * C2
    s1 = np.sum(C1, 0)
    s2 = np.sum(C2, 0)

    s1 = (s1 - np.min(s1)) / (np.max(s1) - np.min(s1))
    s2 = (s2 - np.min(s2)) / (np.max(s2) - np.min(s2))

    s_ours = np.argsort(-(s1+s2))
    return s_ours[:k_num]

# dir_path='/home/CONCOCTION/model/DUAL/data/ours/embedding'
# dir_name = '/home/CONCOCTION/model/DUAL/data/ours/path_text'
# k_num=5
def main(dir_path,dir_name,k_num,CONFIG):

    
    config = ConfigParser()
    config.read(CONFIG)

    # Load Data
    print('Load Data....')
    # train_data = np.load(config['path']['data_path'] + '1fea1.npy')
    # train_label = np.load(config['path']['data_path'] + '1lab1.npy')
    # test_data = np.load(config['path']['data_path'] + '1fea2.npy')
    # test_label = np.load(config['path']['data_path'] + '1lab2.npy')
    

    from tqdm import tqdm
    import os
    import pickle

    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    


    for file in tqdm(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)
        if file.endswith('.pickle'):
            with open(file_path, 'rb') as f:
                map_dict = pickle.load(f)
            merged_tensor = None
            for file_name, value in map_dict.items():
                if isinstance(value, dict):
                    for code, sub_value in value.items():
                        save_name = file_name.split('/')[-1]
                        if len(sub_value) <= 5:
                            f = open(dir_name + '/' + save_name, 'a')
                            f.write('--------------filename--------------' + '\n')
                            f.write(file_name + '\n')
                            f.write('--------------code--------------' + '\n')
                            for line in code:
                                line = line.strip("'")
                                f.write(line + '\n')
                            for path_embedding, path in sub_value.items():
                                f.write('--------------path--------------' + '\n')
                                for line in path:
                                    line = line.strip("'")
                                    line = line.strip()
                                    f.write(line + '\n')
                                f.write('--------------path over--------------' + '\n')
                            f.close()
                            break
                        if isinstance(sub_value, dict):
                            for path_embedding, path in sub_value.items():
                                if merged_tensor is None:
                                    merged_tensor = path_embedding
                                else:
                                    merged_tensor = torch.cat([merged_tensor, path_embedding], dim=0)
                        if merged_tensor is None:
                            continue
                        else:
                            feature=merged_tensor
                            if torch.cuda.is_available():
                                feature = feature.cuda()
                            batch_size = len(feature)
                            # print('Done!')

                            p_q = json.loads(config.get('trainsetting', 'p_q'))
                            p_Rq = json.loads(config.get('trainsetting', 'p_Rq'))

                            # print('Start trainging!')
                            # for i in p_q:
                            #     for j in p_Rq:
                            i = p_q[0]
                            j= p_Rq[0]
                            f = open(config['path']['log_save_path'] + 'results.txt', 'a')
                            parameter = 'Q:' + str(i) + '  ' + 'RQ:' + str(j) + '\n'
                            # f.write(parameter)

                            # print('Inti Net....')
                            net = AutoEncoder(batch_size, config.getint('trainsetting', 'k'),
                                            json.loads(config.get('trainsetting', 'layers')))


                            # pre_dict = torch.load(config['path']['model_load_path'] + 'only_AE.pt')
                            model_dict = net.state_dict()
                            # pre_dict = {k: v for k, v in pre_dict.items() if
                            #             (k in model_dict) and (k not in ['coeff', 'coeff_cluster', 'kmeans'])}
                            # model_dict.update(pre_dict)
                            net.load_state_dict(model_dict)

                            if torch.cuda.is_available():
                                net = net.cuda()
                            # print('Done!')

                            net.eval()
                            train_data = net.encoder(feature).cpu().detach().numpy()
                            # test_data = net.encoder(feature_).cpu().detach().numpy()

                            net = train_AE_withC(net, feature, i, j, config)
                            path_num = ttest_AE_withC(net, train_data, k_num)

                            #


                            f = open(dir_name + '/' + save_name, 'a')
                            f.write('--------------filename--------------' + '\n')
                            f.write(file_name + '\n')
                            f.write('--------------code--------------' + '\n')
                            for line in code:
                                line = line.strip("'")
                                f.write(line + '\n')

                            # path_num=path_num.tolist()
                            #
                            count=0
                            for path_embedding, path in sub_value.items():
                                if count in path_num:
                                    f.write('--------------path--------------' + '\n')
                                    for line in path:
                                        line = line.strip("'")
                                        line = line.strip()
                                        f.write(line + '\n')
                                    f.write('--------------path over--------------' + '\n')
                                count = count + 1
                            f.close()
                            continue
    
    print(f"saved the select path to {dir_name}")







if __name__ == '__main__':
    config=os.path.join(os.path.dirname(__file__),'config/train.config')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, help="data path"
    )
    parser.add_argument(
        "--stored_path", type=str, help="stored data path"
    )
    parser.add_argument(
        "--k_num", type=int, default=5,help="the num of the choosed path"
    )
    parser.add_argument("--config", type=str, help="the config file", default=config)
    args = parser.parse_args()
    data_path=args.data_path
    stored_path=args.stored_path
    k_num=args.k_num
    CONFIG=args.config
    main(data_path,stored_path,k_num,CONFIG)