import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt
from utils import *
from data_loader import *
import IPython
from tqdm import tqdm
from deepod.metrics import tabular_metrics

class Solver(object):
    DEFAULTS = {}   
    def __init__(self, data_loader, config):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader

        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        # Define model
        self.dagmm = DaGMM(n_gmm = self.gmm_k,latent_dim = self.latent_dim,num_states = self.num_states)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)

        # Print networks
        self.print_network(self.dagmm, 'DaGMM')

        if torch.cuda.is_available():
            self.dagmm.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.dagmm.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_dagmm.pth'.format(self.pretrained_model))))

        # print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)

        # print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def reset_grad(self):#重置模型的梯度
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):#将数据移到GPU
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self):
        iters_per_epoch = len(self.data_loader)

        # Start with trained model if exists
        if self.pretrained_model:
            start = int(self.pretrained_model.split('_')[0])
        else:
            start = 0

        # Start training
        iter_ctr = 0
        start_time = time.time()

        self.ap_global_train = np.array([0,0,0])
        for e in range(start, self.num_epochs):
            # print("RSD")
            for i, (input_data, labels) in enumerate(tqdm(self.data_loader)):
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)
                # input_data = input_data.reshape(-1, 1, 4800)

                total_loss,likehood_cw, recon_error, cov_diag = self.dagmm_step(input_data)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                # loss['likehood_cw'] = likehood_cw.item()
                loss['likehood_cw'] = likehood_cw.item() * self.lambda_energy
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item() * self.lambda_cov_diag


                # print(input_data.shape,labels.shape,i,self.log_step,self.model_save_step,"WW")
                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr_tmp = []
                    for param_group in self.optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)

                    IPython.display.clear_output()
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                    else:
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1

                        plt.show()

                    # print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)
                # Save model checkpoints
                if (i+1) % self.model_save_step == 0:
                    torch.save(self.dagmm.state_dict(),
                        os.path.join(self.model_save_path, '{}_{}_dagmm.pth'.format(e+1, i+1)))

    def dagmm_step(self, input_data):
        self.dagmm.train()
        enc, dec, z, gamma, xi = self.dagmm(input_data)

        total_loss, likehood_cw, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, xi, self.lambda_energy, self.lambda_cov_diag)
        # print("BEFORE")
        self.reset_grad()
        total_loss.backward()
        # print("OK")

        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()

        return total_loss,likehood_cw, recon_error, cov_diag

    def test(self):
        print("======================TEST MODE======================")
        self.dagmm.eval()
        self.data_loader.dataset.mode="train"

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        gamma_sum_sum = 0
        xi_sum = 0

        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            # input_data = input_data.reshape(-1, 1, 4800)
            # enc, dec, z, gamma = self.dagmm(input_data)
            # phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)
            # batch_gamma_sum = torch.sum(gamma, dim=0)

            enc, dec, z, gamma, xi = self.dagmm(input_data)
            start_prob, new_transition_matrix, ci, new_mean, new_covariance = self.dagmm.BW_CW(z, gamma, xi)
            batch_gamma_sum = torch.sum(gamma, dim=0) #(I,n_gmm)
            batch_gamma_sum_sum = torch.sum(batch_gamma_sum, dim=-1)  # (I)
            batch_xi_sum = torch.sum(xi, dim=0)  # (I,J)

            gamma_sum += batch_gamma_sum
            gamma_sum_sum += batch_gamma_sum_sum
            xi_sum += batch_xi_sum

            mu_sum += new_mean * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only仅保留分子的和
            cov_sum += new_covariance * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only仅保留分子的和
            # print("it:",it)
            N += input_data.size(0)

        train_new_transition_matrix = xi_sum/gamma_sum_sum.unsqueeze(-1)
        train_ci = gamma_sum / gamma_sum_sum.unsqueeze(-1)
        train_new_mean = mu_sum / gamma_sum.unsqueeze(-1)
        train_new_covariance = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # print("N:",N)
        # print("phi :\n",train_phi)
        # print("mu :\n",train_mu)
        # print("cov :\n",train_cov)

        train_likehood_cw = []
        train_labels = []
        train_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):

            input_data = self.to_var(input_data)
            # input_data = input_data.reshape(-1,1,4800)
            enc, dec, z, gamma, xi = self.dagmm(input_data)
            likehood_cw, cov_diag = self.dagmm.posterior_prob(z, start_prob, train_new_transition_matrix, train_ci, train_new_mean,
                                                        train_new_covariance)

            # 将结果转换为NumPy数组并存储
            # print("likehood_cw",likehood_cw)

            train_likehood_cw.append((likehood_cw).data.cpu().numpy())
            train_z.append(z.data.cpu().numpy())
            train_labels.append(labels.numpy())

        # 将结果堆叠为NumPy数组
        train_likehood_cw = np.concatenate(train_likehood_cw,axis=0)
        train_z = np.concatenate(train_z,axis=0)
        train_labels = np.concatenate(train_labels,axis=0)
        # 指定保存的文件路径
        file_path = 'D:/cw/dagmm-master-myhmm1127/train_likehood_cw.txt'
        # 使用savetxt函数保存数据到txt文件
        np.savetxt(file_path, train_likehood_cw)


        self.data_loader.dataset.mode="test"
        test_likehood_cw = []
        test_labels = []
        test_z = []
        for it, (input_data, labels) in enumerate(self.data_loader):
            input_data = self.to_var(input_data)
            # input_data = input_data.reshape(-1, 1, 4800)
            # enc, dec, z, gamma = self.dagmm(input_data)
            enc, dec, z, gamma, xi = self.dagmm(input_data)
            # likehood_cw, cov_diag = self.dagmm.posterior_prob(z)
            likehood_cw, cov_diag = self.dagmm.posterior_prob(z, start_prob, train_new_transition_matrix, train_ci,train_new_mean,train_new_covariance) #cw
            # sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
            test_likehood_cw.append(likehood_cw.data.cpu().numpy())
            test_z.append(z.data.cpu().numpy())
            test_labels.append(labels.numpy())


        test_likehood_cw = np.concatenate(test_likehood_cw,axis=0)#(150,)
        test_z = np.concatenate(test_z,axis=0)
        test_labels = np.concatenate(test_labels,axis=0)
        # print("train_energy:", train_labels.shape, train_labels)
        # print("test_energy:",test_labels.shape,test_labels)
        combined_likehood_cw = np.concatenate([train_likehood_cw, test_likehood_cw], axis=0)
        # combined_energy = np.concatenate([train_energy], axis=0)#cw
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)
        # 找出合并能量数组中的阈值，这里取合并能量数组中排在前 80% 的值作为阈值
        # thresh = np.percentile(combined_likehood_cw, 100 - 49)#20 #合并能量中（训练集+测试集合）有50%为正常样本，所以取小于50%的为阈值
        # thresh = np.percentile(train_likehood_cw, 5) #表示有x%的数据小于或等于计算得到的thresh 值 #19.26
        thresh =  min(train_likehood_cw) #若实际标签中的0多于预测标签，那么值调小
        # thresh_real = min(train_likehood_cw)
        print("thresh",thresh)
        print("test_energy:",test_likehood_cw.shape,test_likehood_cw)
        # print("Threshold :", thresh)
        # print("Threshold :", thresh_real)
        # # 指定保存的文件路径
        file_path = 'D:/cw/dagmm-master-myhmm1221/combined_likehood_cw.txt'
        # 使用savetxt函数保存数据到txt文件
        np.savetxt(file_path, combined_likehood_cw)

        # # 指定保存的文件路径
        file_path = 'C:/Users/THUNDEROBOT/Desktop/ljs24.4.23/dagmm-master-myhmm1221/test_z.txt'
        # 使用savetxt函数保存数据到txt文件
        np.savetxt(file_path, test_z)


        # 根据阈值对测试能量进行二分类预测，生成预测值和真实值
        # pred = (test_likehood_cw > thresh).astype(int)# 如果测试能量大于阈值，则预测值为1，否则为0
        pred = (test_likehood_cw < thresh).astype(int)  # 如果测试似然小于阈值，则预测值为1，否则为0
        # pred = (test_likehood_cw < thresh_real).astype(int)  # 如果测试似然小于阈值，则预测值为1，否则为0
        print("pred:",pred)
        gt = test_labels.astype(int)# 真实的二分类标签
        print("gt:", gt)
        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')
        auc_con, ap_con, _ = tabular_metrics(gt, pred)
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
        print(f"rate - auc: {auc_con:.4f}, ap: {ap_con:.4f}, F1 Score: {f_score:.4f}")
