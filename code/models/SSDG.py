# -*- coding: utf-8 -*-
import math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import torch.utils.data as Data
from utils import mmd


class DFE(nn.Module):

    def __init__(self,eps=1e-6, alpha=0.1):
        super().__init__()
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)
    def forward(self, x):
        N, C, L = x.shape
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True)
        sig = (var + self.eps).sqrt()
        x_perturbed=x
        if self.training:
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig
            mu_random = torch.empty((N, C, 1), dtype=torch.float32).uniform_(0.5, 1.0).to(x.device)
            var_random = torch.empty((N, C, 1), dtype=torch.float32).uniform_(0.5, 1.0).to(x.device)
            lmda = self.beta.sample((N, C,1))
            bernoulli = torch.bernoulli(lmda).to(x.device)
            mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
            sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
            x_perturbed=x_normed * sig_mix + mu_mix

        return x_perturbed
class SSDGClassifier(nn.Module):
    def __init__(self, num_features, num_classes, LR=False, method="M1"):
        super().__init__()
        self.set=method
        if LR:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features // 2)
        self.p2 = nn.Linear(num_features // 2, num_features// 4 )
        self.p3 = nn.Linear(num_features // 4, num_features // 8)
        self.p4 = nn.Linear(num_features // 4, num_features // 2)
        self.p5 = nn.Linear(num_features // 2, num_features)
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))

        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)


    def forward(self, x, LR=False, noise=False):
        if LR and  self.set=="M4":
            x_mean= x.mean(dim=0, keepdim=True)
            x_mean = torch.relu(self.p1( x_mean))
            x_mean = torch.relu(self.p2(x_mean))
            x_mean = torch.relu(self.p3(x_mean))
            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device)
                x_mean = torch.cat((x_mean, noise), dim=1)
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1)
            x_mean =torch.relu(self.p4(x_mean))
            x_mask = torch.relu(self.p5(x_mean))
            x1 = self.h1(x_mask)
            x2 = self.h2(x_mask)
            a=torch.matmul(x2.t(), x1)
            w_mask = torch.sigmoid(a)
            self.w_new = self.w * w_mask
            return torch.matmul(x, self.w_new.t())
        else:
            return torch.matmul(x, self.w.t())
class SSDGFea_Extraction(nn.Module):
    def __init__(self,in_channel=1,method=""):
        super().__init__()
        self.set=method
        self.DFE= DFE()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=1),
            nn.InstanceNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, stride=1),
            nn.InstanceNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.avgp1 = nn.AdaptiveAvgPool1d(1)
    def  forward(self, x,trid=False):
        x1 = self.layer1(x)
        if trid and (self.set=="M4"):
            x2= self.DFE(x1)
        x = self.layer2(x2)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        out = x.view(x.size(0), -1)

        return out
class SSDG(nn.Module):
    def __init__(self, in_channel=1, num_classes=5, lr=0.01, set="M4"):
        super(SSDG, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.set_trid = set
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.G = SSDGFea_Extraction(in_channel=in_channel, method=self.set_trid).to(self.device)
        self.C = SSDGClassifier(1024, num_classes, LR=True, method=self.set_trid).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.u_loss = nn.CrossEntropyLoss(reduction="none").to(self.device)
        self.classwise_acc_1 = torch.ones((self.num_classes,)).to(self.device)
        self.classwise_acc_2 = torch.ones((self.num_classes,)).to(self.device)
        self.optimizer = optim.Adam(
            [{'params': self.G.parameters(), 'lr': 1e-4},
             {'params': self.C.parameters(), 'lr': 1e-4}],
            weight_decay=1e-4
        )



    def forward(self, TR_dataloader, epoch_it):
        self.G.train()
        self.C.train()
        epoch_s_loss = 0.0
        epoch_u_loss = 0.0
        loss_all = 0.0
        loss_sa_all=0.0
        y_u_pred_list = []
        y_u_true_list = []
        mask_u_list = []

        for domain, dataset in enumerate(TR_dataloader):
            if domain == 0:
                train_loader_x_src_0 = Data.DataLoader(dataset, batch_size=128)
                selected_label_0 = torch.ones((len(dataset),), dtype=torch.long) * -1
            elif domain == 1:
                train_loader_x_src_1 = Data.DataLoader(dataset, batch_size=128)
                selected_label_1 = torch.ones((len(dataset),), dtype=torch.long) * -1
                selected_label_1 = selected_label_1.to(self.device)
            elif domain == 2:
                train_loader_x_src_2 = Data.DataLoader(dataset, batch_size=128)
                selected_label_2 = torch.ones((len(dataset),), dtype=torch.long) * -1
                selected_label_2 = selected_label_2.to(self.device)

        threshold_1_list=[]
        threshold_2_list=[]
        for batch_src_0, batch_src_1, batch_src_2 in zip(train_loader_x_src_0, train_loader_x_src_1, train_loader_x_src_2  ):
            batch_x_0, batch_y_0, batch_domain_0, _ = batch_src_0  # Labeled data from domain 0
            batch_x_1, batch_y_1, batch_domain_1, x_index_1 = batch_src_1  # Unlabeled data from domain 1
            batch_x_2, batch_y_2, batch_domain_2, x_index_2 = batch_src_2  # Unlabeled data from domain 2

            labels = torch.cat([batch_y_1, batch_y_2])
            unsup_warmup = np.clip(epoch_it / (0.8*100), a_min=0.0, a_max=1.0)

            self.optimizer.zero_grad()
            with torch.no_grad():
                f_x_1 = self.G(batch_x_1.to(self.device), trid=True)
                z_xu_k_1 = self.C(f_x_1, LR=True)
                p_xu_1 = F.softmax(z_xu_k_1, dim=1)
                p_xu_maxval_1, y_xu_pre_1 = p_xu_1.max(dim=1)
                threshold_1 = 0.95 * (
                        self.classwise_acc_1[y_xu_pre_1] / (2.0 - self.classwise_acc_1[y_xu_pre_1])
                )
                mask_x_1 = (p_xu_maxval_1 >= threshold_1).float()
                threshold_1_list.append(threshold_1.cpu())

                f_x_2= self.G( batch_x_2.to(self.device), trid=True)
                z_xu_k_2 = self.C(f_x_2, LR=True)
                p_xu_2 = F.softmax(z_xu_k_2, dim=1)
                p_xu_maxval_2, y_xu_pre_2 = p_xu_2.max(dim=1)
                threshold_2 = 0.95 * (
                        self.classwise_acc_2[y_xu_pre_2] / (2.0 - self.classwise_acc_2[y_xu_pre_2])
                )
                mask_x_2 = (p_xu_maxval_2 >= threshold_2).float()
                threshold_2_list.append(threshold_2.cpu())


                mask_xu = torch.cat([mask_x_1, mask_x_2])
                y_xu_pred = torch.cat([y_xu_pre_1, y_xu_pre_2])
                y_u_pred_list.append(y_xu_pred.clone().cpu().detach().numpy())
                y_u_true_list.append(labels.clone().cpu().detach().numpy())
                mask_u_list.append(mask_xu.clone().cpu().detach().numpy())

                if self.set_trid in ["M4"]:
                    selected_indices_1 = torch.where(mask_x_1 == 1)[0]
                    if selected_indices_1.nelement() != 0:
                        pseudo_labels_1 = y_xu_pre_1[selected_indices_1]
                        selected_label_1[selected_indices_1] = pseudo_labels_1
                        pseudo_counter_1 = Counter(pseudo_labels_1.tolist())
                        if max(pseudo_counter_1.values()) < len(selected_label_1):
                            for i in range(self.num_classes):
                                self.classwise_acc_1[i] = (
                                    pseudo_counter_1.get(i, 0) / max(pseudo_counter_1.values())
                                )

                    selected_indices_2 = torch.where(mask_x_2 == 1)[0]
                    if selected_indices_2.nelement() != 0:
                        pseudo_labels_2 = y_xu_pre_2[selected_indices_2]
                        selected_label_2[selected_indices_2] = pseudo_labels_2
                        pseudo_counter_2 = Counter(pseudo_labels_2.tolist())
                        if max(pseudo_counter_2.values()) < len(selected_label_2):
                            for i in range(self.num_classes):
                                self.classwise_acc_2[i] = (
                                    pseudo_counter_2.get(i, 0) / max(pseudo_counter_2.values())
                                )


            f_k_0= self.G(batch_x_0.to(self.device),trid=True)
            z_k_0 = self.C(f_k_0, LR=True,noise=True)
            loss_s = self.criterion(z_k_0, batch_y_0.to(self.device))
            f_xu_k_aug_1 = self.G(batch_x_1.clone().to(self.device),trid=True)
            z_xu_k_aug_1 = self.C(f_xu_k_aug_1, LR=True,noise=True)
            loss_u_1 = self.u_loss(z_xu_k_aug_1, y_xu_pre_1)
            loss_u_1_m = (loss_u_1 * mask_x_1).mean()
            f_xu_k_aug_2 = self.G(batch_x_2.clone().to(self.device),trid=True)
            z_xu_k_aug_2 = self.C(f_xu_k_aug_2, LR=True,noise=True)
            loss_u_2 = self.u_loss(z_xu_k_aug_2, y_xu_pre_2)
            loss_u_2_m = (loss_u_2 * mask_x_2).mean()


            if self.set_trid in["M4"] and epoch_it>50:
                class_prototypes = {}
                features_0 = f_k_0.detach()
                prototypes_0 = []
                for c in range(self.num_classes):
                    class_features = features_0[batch_y_0 == c]
                    if class_features.size(0) > 0:
                        prototype = class_features.mean(dim=0)
                    else:
                        prototype = torch.zeros(features_0.size(1)).to(self.device)
                    prototypes_0.append(prototype)
                class_prototypes[0] = torch.stack(prototypes_0)


                features_1 = f_xu_k_aug_1.detach()
                prototypes_1 = []
                for c in range(self.num_classes):
                    class_features = features_1[y_xu_pre_1 == c]
                    if class_features.size(0) > 0:
                        prototype = class_features.mean(dim=0)
                    else:
                        prototype = torch.zeros(features_1.size(1)).to(self.device)
                    prototypes_1.append(prototype)
                class_prototypes[1] = torch.stack(prototypes_1)


                features_2 = f_xu_k_aug_2.detach()
                prototypes_2 = []
                for c in range(self.num_classes):
                    class_features = features_2[y_xu_pre_2 == c]
                    if class_features.size(0) > 0:
                        prototype = class_features.mean(dim=0)
                    else:
                        prototype = torch.zeros(features_2.size(1)).to(self.device)
                    prototypes_2.append(prototype)
                class_prototypes[2] = torch.stack(prototypes_2)

                prototypes_0_norm = F.normalize(class_prototypes[0], p=2, dim=1)
                f_u_0 = F.normalize(f_k_0, p=2, dim=1)
                similarity_0 = torch.mm(f_u_0, prototypes_0_norm.t())
                mask_x_0 = torch.ones(f_u_0.size(0)).to(self.device)
                f_u_1 = F.normalize(f_xu_k_aug_1, p=2, dim=1)
                similarity_1 = torch.mm(f_u_1,prototypes_0_norm.t())
                f_u_2 = F.normalize(f_xu_k_aug_2, p=2, dim=1)
                similarity_2 = torch.mm(f_u_2, prototypes_0_norm.t())

                loss_sa_0 = self.compute_sa_loss_multi(
                    similarity_0,  mask_x_0, batch_y_0
                )
                sim_top1_0, idx = similarity_0.topk(k=1, dim=1, sorted=True)
                loss_sim_0 = ((1 - sim_top1_0.squeeze(1)) * mask_x_0).mean()

                loss_sa_1 = self.compute_sa_loss_multi(
                    similarity_1,  mask_x_1, y_xu_pre_1
                )
                sim_top1_1, idx = similarity_1.topk(k=1, dim=1, sorted=True)
                loss_sim_1 = ((1 - sim_top1_1.squeeze(1)) * mask_x_1).mean()

                loss_sa_2 = self.compute_sa_loss_multi(
                    similarity_2,  mask_x_2, y_xu_pre_2
                )
                sim_top1_2, idx = similarity_2.topk(k=1, dim=1, sorted=True)
                loss_sim_2 = ((1 - sim_top1_2.squeeze(1))* mask_x_2).mean()



                loss_sa = loss_sa_0 + loss_sa_1 + loss_sa_2
                MMD_loss = mmd.mmd_rbf_noaccelerate(z_k_0, z_xu_k_aug_1) + mmd.mmd_rbf_noaccelerate(z_k_0, z_xu_k_aug_2)
                loss_sim = loss_sim_0 + loss_sim_1 + loss_sim_2
            else:
                loss_sa=0.0
                loss_sim=0.0
                MMD_loss=0.0


            loss = (loss_u_1_m +loss_u_2_m) * unsup_warmup +loss_s+loss_sa*0.5+loss_sim*15+MMD_loss
            loss.backward()
            self.optimizer.step()
            epoch_s_loss += loss_s.item()
            epoch_u_loss += loss_u_1.mean().item()+loss_u_2.mean().item()
            loss_all += loss.item()
            loss_sa_all+=loss_sim



        return loss_all

    def compute_sa_loss_multi(self, similarity, mask_x, num_classes):

        loss_sim = F.cross_entropy(similarity,num_classes.to(self.device), reduction="none")*3
        loss_sim = (loss_sim * mask_x).mean()
        return loss_sim

    def model_inference(self, input):
        self.G.eval()
        self.C.eval()
        with torch.no_grad():
            features = self.G(input.to(self.device),trid=True)
            prediction = self.C(features, LR=False, noise=False)
        return prediction






