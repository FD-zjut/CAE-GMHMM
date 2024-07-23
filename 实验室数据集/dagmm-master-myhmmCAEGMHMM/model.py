import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
from utils import *
# from math import pi

def Laplace(p):
    A = 0.08
    ep = 0.03
    tal = 0.1
    f = 50
    w = 2 * np.pi * f
    q = torch.tensor(1 - pow(ep, 2))
    y = A * torch.exp((-ep / (torch.sqrt(q))) * (w * (p - tal))) * (-torch.sin(w * (p - tal)))

    return y

class Laplace_fast(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):
        super(Laplace_fast, self).__init__()
        if in_channels != 1:
            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):

        time_disc = torch.linspace(0, 1, steps=int((self.kernel_size)))
        p1 = time_disc.cuda() - self.b_.cuda() / self.a_.cuda()
        laplace_filter = Laplace(p1)
        self.filters = (laplace_filter).view(self.out_channels, 1, self.kernel_size).cuda()

        return F.conv1d(waveforms, self.filters, stride=1, padding=2, dilation=1, bias=None, groups=1)

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        # l = torch.potrf(a, False)
        # l = torch.cholesky(a, False)#cw
        l = torch.linalg.cholesky(a)#cw # 使用 torch.linalg 提供的 Cholesky 分解函数进行计算
        ctx.save_for_backward(l) # 保存 l 以备后向传播使用
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables # 从保存的变量中获取 l
        linv = l.inverse() # 计算 l 的逆矩阵
        # 计算用于反向传播的中间变量
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv)) # 计算反向传播的梯度
        return s
    
class DaGMM(nn.Module):
    """Residual Block."""
    def __init__(self, n_gmm = 4, latent_dim=7,num_states=6): #latent_dim为原始特征数，num_states为状态数
        super(DaGMM, self).__init__()
        #卷积 卷积核大小为3，步幅为1，填充为1
        self.layer2 = nn.Sequential(
            nn.Conv1d(6, 12, kernel_size=3, stride=1, padding=1),
            #归一化层
            nn.BatchNorm1d(12),
            #激活函数
            nn.ReLU(),
            #池化层
            nn.MaxPool1d(2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2))

        #特征提取和降维
        self.Linear1 = nn.Linear(7200, 150)#6*window/2;  window = window/2/2/2 (192=3*128/2) #600 6*1600/2
        self.Linear2 = nn.Linear(150, 100)
        self.Linear3 = nn.Linear(100, 50)
        self.Linear4 = nn.Linear(50, 30)
        self.Linear5 = nn.Linear(30, 10)
        self.Linear6 = nn.Linear(10, latent_dim-6*2)
        #重构出原始的高维特征向量
        self.fc1 = nn.Linear(latent_dim-6*2, 10)  # map z_c to a higher dimensional space
        self.fc2 = nn.Linear(10, 30)
        self.fc3 = nn.Linear(30, 50)
        self.fc4 = nn.Linear(50, 100)
        self.fc5 = nn.Linear(100, 150)
        self.fc6 = nn.Linear(150, 7200)
        #进行反卷积
        self.layer1_d = nn.Sequential(
            nn.ConvTranspose1d(24, 24, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU())
        self.layer2_d = nn.Sequential(
            nn.ConvTranspose1d(24, 12, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(12),
            nn.ReLU())
        self.layer3_d = nn.Sequential(
            nn.ConvTranspose1d(12, 6, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(6),
            nn.ReLU())
        #每个神经元有 50% 的概率被随机丢弃，防止过拟合，提高模型的泛化能力。
        self.dropout = nn.Dropout(p=0.5)

        #定义了两个神经网络模型
        layers = []
        layers += [nn.Linear(latent_dim, 20)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(20, num_states ** 2)]
        layers += [nn.Softmax(dim=1)]
        self.estimation1 = nn.Sequential(*layers)  # 论文里的xi

        layers = []
        layers += [nn.Linear(latent_dim, 20)]
        layers += [nn.Tanh()]
        layers += [nn.Dropout(p=0.5)]
        layers += [nn.Linear(20, num_states * n_gmm)]
        layers += [nn.Softmax(dim=1)]
        self.estimation2 = nn.Sequential(*layers)  # 论文里的gamma

        #注册了一些张量缓冲区，这些缓冲区的内容不会被更新，而是作为模型的固定参数。
        self.register_buffer("start_prob", torch.ones(num_states) / num_states)
        self.register_buffer("new_transition_matrix", torch.zeros(num_states, num_states))
        self.register_buffer("ci", torch.zeros(n_gmm, latent_dim))
        self.register_buffer("new_mean", torch.zeros(num_states, n_gmm, latent_dim))
        self.register_buffer("new_covariance", torch.zeros(num_states, n_gmm, latent_dim, latent_dim))

        # self.latent_dim = latent_dim
        self.num_states = num_states
        self.n_gmm = n_gmm

    #定义了一个编码器
    def encoder(self, x):
        # h = self.layer1(x)
        h = self.layer2(x)
        h = self.layer3(h)
        h = self.layer4(h)
        h = h.view(h.size(0), -1)
        h = self.Linear1(h)
        h = nn.ReLU()(h)
        h = self.Linear2(h)
        h = nn.ReLU()(h)
        h = self.Linear3(h)
        h = nn.ReLU()(h)
        h = self.Linear4(h)
        h = nn.ReLU()(h)
        h = self.Linear5(h)
        h = nn.ReLU()(h)
        h = self.Linear6(h)
        h = nn.ReLU()(h)
        # h = self.dropout(h)
        return h

    #定义了一个解码器
    def decoder(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        # print(x.shape)
        x = x.reshape(5, 24, -1) #12通道数
        h = self.layer1_d(x)
        h = self.layer2_d(h)
        h = self.layer3_d(h)
        # h = self.layer4_d(h)
        return h

    #计算欧氏距离
    def relative_euclidean_distance(self, a, b):
        return (a-b).norm(2, dim=2) / a.norm(2, dim=2)

    #前向算法
    def forward(self, x):

        enc = self.encoder(x)#编码

        dec = self.decoder(enc)#解码

        rec_cosine = F.cosine_similarity(x, dec, dim=2)#余弦相似性
        rec_euclidean = self.relative_euclidean_distance(x, dec)#相对欧几里得距离
        #unsqueeze(-1)表示在最后一个维度（最内层维度）上增加一个维度
        z = torch.cat([enc, rec_euclidean, rec_cosine], dim=1)#拼接
        # z = torch.cat([enc], dim=1)  # 拼接 cw
        D1wei, D2wei = z.size()  # z(5,3) 3是拼接. D1wei是时间，D2wei是latent_dim
        xiyuan = self.estimation1(z)

        # 使用view函数将其转换为形状为（5，3，3）的张量
        xihou = xiyuan.view(D1wei, self.num_states, self.num_states) #3是num_states状态数量.4是n-gmm数量
        # 提取第一维度的前四个元素，并保持其他维度不变
        xi = xihou[: D1wei - 1 ,:,:] #（4，3，3）(T-1,I,J)
        gammayuan = self.estimation2(z)#(5,3) (T,I)
        gamma = gammayuan.view(D1wei, self.num_states, self.n_gmm)  # 3是num_states状态数量.4是n-gmm数量
        return enc, dec, z, gamma, xi

    #计算了混合高斯模型 (GMM) 的参数
    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0) #原先的代码，不用管
        #对每个混合高斯成分，计算所有样本属于该成分的概率之和
        sum_gamma = torch.sum(gamma, dim=0)#dim=0 表示沿着第一个维度（通常是行）进行操作，如求和
        #计算每个混合高斯成分的先验概率
        phi = (sum_gamma / N)
        self.phi = phi.data

        #计算了混合高斯模型的均值 mu 和协方差矩阵 cov
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        #计算样本与均值之间的差值
        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim = 0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov

    #实现了混合高斯模型的 Baum-Welch (BW) 和 Cover-Wolfowitz (CW) 算法，用于参数的更新和优化。
    def BW_CW(self, z , gamma, xi):

        start_prob = torch.sum(gamma[0, :], dim=1) #（1,I）
        sum_gamma = torch.sum(gamma, dim=0)#按第一个维度时间T求和
        new_transition_matrix = torch.sum(xi, dim=0) / ((torch.sum(sum_gamma, dim=1)).unsqueeze(-1)) # （I,J）
        # 计算每个混合高斯成分的权重
        ci = sum_gamma/((torch.sum(sum_gamma, dim=1)).unsqueeze(-1))
        # 计算每个混合高斯成分的均值 new_mean
        new_mean = torch.sum(gamma.unsqueeze(-1) * (z.unsqueeze(1).unsqueeze(1)), dim=0) / sum_gamma.unsqueeze(-1)#（I,n_gmm,latent_dim）
        z_new_mean = (z.unsqueeze(1).unsqueeze(1) - new_mean.unsqueeze(0))
        z_new_mean_outer = z_new_mean.unsqueeze(-1) * z_new_mean.unsqueeze(-2)
        # 计算每个混合高斯成分的协方差矩阵
        new_covariance = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_new_mean_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.start_prob = start_prob.data
        self.new_transition_matrix = new_transition_matrix.data
        self.ci = ci.data
        self.new_mean = new_mean.data
        self.new_covariance = new_covariance.data
        return start_prob, new_transition_matrix, ci, new_mean, new_covariance

    #计算样本的能量值
    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None: #原先的代码，不用管
            phi = to_var(self.phi)
        if mu is None:
            mu = to_var(self.mu)
        if cov is None:
            cov = to_var(self.cov)

        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        cov_inverse = []  # 用于存储协方差矩阵的逆矩阵
        det_cov = []  # 用于存储协方差矩阵的行列式
        cov_diag = 0
        eps = 1e-6
        for i in range(k):
            # K x D x D
            cov_k = cov[i] + to_var(torch.eye(D)*eps) # 将矩阵对角线上加上一个小的正数，避免矩阵奇异性
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0)) # 计算协方差矩阵的逆矩阵并存储*******
            # 使用 Cholesky 分解计算协方差矩阵的行列式，这里涉及到了上述提到的 Cholesky 类
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag()) # 计算协方差矩阵对角线元素的倒数之和*****

        # K x D x D
        cov_inverse = torch.cat(cov_inverse, dim=0)
        # K
        det_cov = torch.cat(det_cov).cuda()
        #det_cov = to_var(torch.from_numpy(np.float32(np.array(det_cov))))

        # N x K
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)#E(z)的分子
        # for stability (logsumexp)
        max_val = torch.max((exp_term_tmp).clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)

        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (det_cov).unsqueeze(0), dim = 1) + eps)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim = 1) + eps)
        # sample_energy = -max_val.squeeze() - torch.log(torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt((2*np.pi)**D * det_cov)).unsqueeze(0), dim = 1) + eps)

        # print("sample_energy",sample_energy.shape,sample_energy)
        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    #计算了观测值的发射概率，用于隐马尔可夫模型的预测步骤中
    def compute_emission_prob(self, z, ci, new_mean, new_covariance):
        T_cw,_ = z.size()
        v, k, D, _ = new_covariance.size()
        # z_new_mean = (z.unsqueeze(1).unsqueeze(1) - new_mean.unsqueeze(0))

        # cov_inverse = []  # 用于存储协方差矩阵的逆矩阵
        # det_cov = []  # 用于存储协方差矩阵的行列式
        cov_diag = 0
        eps = 1e-6
        cov_inverse = torch.empty(v, k, D, D)
        det_cov = torch.empty(v, k)
        # diagonal_index = min(new_covariance.size(2), new_covariance.size(3))
        # new_covariance[:, :, range(diagonal_index), range(diagonal_index)] += eps # 将矩阵对角线上加上一个小的正数，避免矩阵奇异性
        for i in range(v):
            for j in range(k):
                # print("new_covariance[i,j]",new_covariance[i,j],(new_covariance[i,j]).shape)
                cov_k = new_covariance[i,j] + to_var(torch.eye(D) * eps)  # 将矩阵对角线上加上一个小的正数，避免矩阵奇异性

                cov_inverse[i, j] = torch.inverse(cov_k) #计算协方差矩阵的逆矩阵并存储
                det_cov[i, j] = Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()
                # cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))  # 计算协方差矩阵的逆矩阵并存储*******
                # print("cov_inverse", cov_inverse, "sdfgsdg")
                cov_diag = cov_diag + torch.sum(1 / cov_k.diag())  # 计算协方差矩阵对角线元素的倒数之和*****
        # print("cov_inverse1", cov_inverse,cov_inverse.shape, "dhdgdg")
        # print("cov_diag", cov_diag, cov_diag.shape, "dhdgdg")
        # print("det_cov", det_cov, det_cov.shape, "dhdgdg")
        # V x K x D x D
        covariance_inv = to_var(cov_inverse)

        z_mu = z.unsqueeze(1).unsqueeze(1) - new_mean.unsqueeze(0)
        # print("——————diff", diff)
        # print("observation",diff.size,diff,"qqq",covariance_inv.size,covariance_inv)
        # print("zz",torch.sum(diff @ covariance_inv * diff, dim=1).type,torch.sum(diff @ covariance_inv * diff, dim=1))
        # exponent = -0.5 * torch.sum(diff @ covariance_inv * diff, dim=1)
        exponent = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * covariance_inv.unsqueeze(0), dim=-2) * z_mu, dim=-1)  # E(z)的分子
        # for stability (logsumexp)
        max_val = torch.max((exponent).clamp(min=0), dim=2, keepdim=True)[0]

        exp_term = torch.exp(exponent - max_val)

        emission_prob = max_val.view(T_cw, v) + torch.log(
            torch.sum(ci.unsqueeze(0) * exp_term / (torch.sqrt(to_var(det_cov))).unsqueeze(0), dim=2) + eps) #(5,3) 这里取log可能有问题？
        # emission_prob = max_val.view(T_cw, v)
        # + (torch.sum(ci.unsqueeze(0) * exp_term / (torch.sqrt(to_var(det_cov))).unsqueeze(0), dim=2) + eps) #(5,3) 这里取log可能有问题？
        return emission_prob,cov_diag

#实现了隐马尔可夫模型 (HMM) 的前向算法
    def hmm_forward(self, z , start_prob, new_transition_matrix, ci, new_mean, new_covariance):
        T_cw, _ = z.size()
        v, _ = new_transition_matrix.size()
        for_alpha = to_var(torch.zeros((T_cw, v)))
        emission_prob, cov_diag = self.compute_emission_prob(z, ci, new_mean, new_covariance)
        _temp = emission_prob[0,:] * start_prob
        for_alpha[0,:] += _temp
        for t in range(1, T_cw):

            _temp = torch.sum(_temp.unsqueeze(-1) * new_transition_matrix, dim=1) * emission_prob[t, :] #for_alpha[t-1,:].unsqueeze(-1)取(-1)还是(0)
            for_alpha[t, :] += _temp
            #for_alpha[t, :] += torch.sum(for_alpha[t - 1, :].unsqueeze(-1) * new_transition_matrix, dim=1) * emission_prob[t, :]
        return for_alpha,cov_diag
    #实现了隐马尔可夫模型 (HMM) 的后向算法
    def hmm_backward(self, z , start_prob, new_transition_matrix, ci, new_mean, new_covariance):
        T_cw, _ = z.size()
        v, _ = new_transition_matrix.size()
        back_beta = to_var(torch.zeros((T_cw, v)))
        emission_prob, cov_diag = self.compute_emission_prob(z, ci, new_mean, new_covariance)
        _temp = torch.ones(self.num_states).cuda()
        back_beta[-1, :] += _temp
        # back_beta[-1, :] += torch.ones(3).cuda()
        for t in reversed(range(T_cw - 1)):

            _temp = torch.sum((emission_prob[t+1, :]).unsqueeze(0) * new_transition_matrix * _temp.unsqueeze(0),dim=1)
            back_beta[t, :] += _temp
            # back_beta[t, :] += torch.sum((emission_prob[t+1, :]).unsqueeze(0) * new_transition_matrix * (back_beta[t + 1, :]).unsqueeze(0),dim=1)
        return back_beta, cov_diag

    #优化函数
    def posterior_prob(self, z , start_prob=None, new_transition_matrix=None, ci=None, new_mean=None, new_covariance=None):
        if start_prob is None:
            start_prob = to_var(self.start_prob)
        if new_transition_matrix is None:
            new_transition_matrix = to_var(self.new_transition_matrix)
        if ci is None:
            cov = to_var(self.ci)
        if new_mean is None:
            new_mean = to_var(self.new_mean)
        if new_covariance is None:
            new_covariance = to_var(self.new_covariance)

        for_alpha, cov_diag1 = self.hmm_forward(z , start_prob, new_transition_matrix, ci, new_mean, new_covariance)
        back_beta, cov_diag2 = self.hmm_backward(z , start_prob, new_transition_matrix, ci, new_mean, new_covariance)
        likehood_cw = torch.sum(for_alpha * back_beta,dim=1)
        cov_diag = cov_diag1 + cov_diag2
        return likehood_cw, cov_diag

    #损失函数
    def loss_function(self, x, x_hat, z, gamma, xi, lambda_energy, lambda_cov_diag):

        recon_error = torch.mean((x - x_hat) ** 2)



        start_prob, new_transition_matrix, ci, new_mean, new_covariance = self.BW_CW(z , gamma, xi)

        likehood_cw, cov_diag = self.posterior_prob(z , start_prob, new_transition_matrix, ci, new_mean, new_covariance)

        loss = recon_error - lambda_energy * torch.mean(likehood_cw) + lambda_cov_diag * cov_diag

        return loss, torch.mean(likehood_cw), recon_error, cov_diag
