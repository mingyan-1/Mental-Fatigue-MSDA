# -*- coding:utf-8 -*-
"""
作者：
日期：2022年10月13日
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from utils import *
import data_load
from model import *
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from sklearn import manifold
import msda
import scipy.io as scio
import os
import warnings

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))

def softmax_loss_all_domain(output_s, gt_s, args):
    criterion = nn.CrossEntropyLoss().cuda()
    loss_s = 0
    for domain in range(0, output_s.shape[2]):
        loss_s = loss_s + criterion(output_s[:, :, domain], gt_s[:, domain].long().squeeze())
    return loss_s
def loss_all_domain(data_erp_s, data_ECG_s, data_EMG_s, data_erp_t, data_ECG_t, data_EMG_t, gt_s, G2, G3, G4, F1, F2, batch_size,args):
    output_s = torch.from_numpy(np.zeros((batch_size, 3, 8, data_erp_s.shape[3])))
    out_s_c1 = torch.from_numpy(np.zeros((batch_size, 2, data_erp_s.shape[3])))
    out_s_c2 = torch.from_numpy(np.zeros((batch_size, 2, data_erp_s.shape[3])))
    if args.cuda:
        output_s = output_s.cuda()
        out_s_c1 = out_s_c1.cuda()
        out_s_c2 = out_s_c2.cuda()
    for domain in range(0, data_erp_s.shape[3]):
        erp_s = G2(data_erp_s[:, :, :, domain])
        ECG_s = G3(data_ECG_s[:, :, domain])
        EMG_s = G4(data_EMG_s[:, :, domain])
        output_s[:, :, :, domain] = torch.stack((erp_s, ECG_s, EMG_s), 1)
        out_s_c1[:, :, domain], flatten_data = F1(output_s[:, :, :, domain].float())
        out_s_c2[:, :, domain], flatten_data = F2(output_s[:, :, :, domain].float())
    erp_t = G2(data_erp_t)
    ECG_t = G3(data_ECG_t)
    EMG_t = G4(data_EMG_t)
    output_t = torch.stack((erp_t, ECG_t, EMG_t), 1)
    out_t_c1, flatten_data = F1(output_t)
    out_t_c2, flatten_data = F2(output_t)
    loss_msda = 0.0005 * msda.msda_regulizer(output_s, output_t, batch_size, 5)
    loss_s_c1 = softmax_loss_all_domain(out_s_c1, gt_s, args)
    loss_s_c2 = softmax_loss_all_domain(out_s_c2, gt_s, args)
    return loss_s_c1, loss_s_c2, loss_msda, G2, G3, G4, F1, F2, out_t_c1, out_t_c2


def train_totalmix(num_epoch, source_totalmix_loader, target_test_totalmix_loader, source_test_totalmix_loader, G2, G3, G4, F1, F2,
                   optimizer_g2, optimizer_g3, optimizer_g4, optimizer_f, batch_size, num_k, writer, fold, args):
    value_max = 0
    criterion = nn.CrossEntropyLoss().cuda()
    for ep in range(num_epoch):
        G2.train()
        G3.train()
        G4.train()
        F1.train()
        F2.train()
        for batch_idx, (
        data_erp_s, gt_s, data_erp_t, gt_t, data_ECG_s, data_ECG_t, data_EMG_s,
        data_EMG_t) in enumerate(source_totalmix_loader):
            if args.cuda:
                data_erp_s, gt_s = data_erp_s.cuda(), gt_s.cuda()
                data_erp_t, gt_t = data_erp_t.cuda(), gt_t.cuda()
                data_ECG_s, gt_s = data_ECG_s.cuda(), gt_s.cuda()
                data_ECG_t, gt_t = data_ECG_t.cuda(), gt_t.cuda()
                data_EMG_s, gt_s = data_EMG_s.cuda(), gt_s.cuda()
                data_EMG_t, gt_t = data_EMG_t.cuda(), gt_t.cuda()
            # when pretraining network source only
            eta = 1.0
            data_erp_s = Variable(data_erp_s)
            data_ECG_s = Variable(data_ECG_s)
            data_EMG_s = Variable(data_EMG_s)
            data_erp_t = Variable(data_erp_t)
            data_ECG_t = Variable(data_ECG_t)
            data_EMG_t = Variable(data_EMG_t)
            gt_s = Variable(gt_s)
            # Step A train all networks to minimize loss on source
            optimizer_g2.zero_grad()
            optimizer_g3.zero_grad()
            optimizer_g4.zero_grad()
            optimizer_f.zero_grad()
            loss_s_c1, loss_s_c2, loss_msda, G2, G3, G4, F1, F2, output_t1, output_t2 = loss_all_domain(data_erp_s, data_ECG_s, data_EMG_s, data_erp_t, data_ECG_t, data_EMG_t, gt_s, G2, G3, G4, F1, F2, batch_size, args)
            all_loss = loss_s_c1 + loss_s_c2 + loss_msda
            with torch.autograd.detect_anomaly():
                all_loss.backward()
            optimizer_g2.step()
            optimizer_g3.step()
            optimizer_g4.step()
            optimizer_f.step()

            # Step B train classifier to maximize discrepancy
            optimizer_g2.zero_grad()
            optimizer_g3.zero_grad()
            optimizer_g4.zero_grad()
            optimizer_f.zero_grad()

            loss_s_c1, loss_s_c2, loss_msda, G2, G3, G4, F1, F2, output_t1, output_t2 = loss_all_domain(data_erp_s, data_ECG_s, data_EMG_s, data_erp_t, data_ECG_t, data_EMG_t, gt_s, G2, G3, G4, F1, F2, batch_size, args)
            loss_s = loss_s_c1 + loss_s_c2  + loss_msda
            loss_dis = discrepancy(output_t1, output_t2)
            # entropy_loss = - torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
            # entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
            loss = loss_s.double() - loss_dis.double()# + 0.01 * entropy_loss.double()
            loss.backward()
            optimizer_f.step()
            # Step C train genrator to minimize discrepancy
            for i in range(num_k):
                optimizer_g2.zero_grad()
                optimizer_g3.zero_grad()
                optimizer_g4.zero_grad()
                loss_s_c1, loss_s_c2, loss_msda, G2, G3, G4, F1, F2, output_t1, output_t2 = loss_all_domain(data_erp_s, data_ECG_s, data_EMG_s, data_erp_t, data_ECG_t, data_EMG_t, gt_s, G2, G3, G4, F1, F2, batch_size, args)
                loss_dis = discrepancy(output_t1, output_t2)
                loss1 = loss_s_c1
                loss2 = loss_s_c2
                entropy_loss = -torch.mean(torch.log(torch.mean(output_t1, 0) + 1e-6))
                entropy_loss -= torch.mean(torch.log(torch.mean(output_t2, 0) + 1e-6))
                loss_dis.backward()
                optimizer_g2.step()
                optimizer_g3.step()
                optimizer_g4.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t Dis: {:.6f} msda: {:.6f}'.format(
                        ep, ep, args.epochs,
                        100. * (ep + 1) / args.epochs, loss1.item(), loss2.item(), loss_dis.item(),
                        loss_msda.item()))
                writer.add_scalar('loss_s1', torch.tensor(loss1.item()) / args.log_interval, global_step=ep)
                writer.add_scalar('loss_s2', torch.tensor(loss2.item()) / args.log_interval, global_step=ep)
                writer.add_scalar('loss_dis', torch.tensor(loss_dis.item()) / args.log_interval, global_step=ep)
            if batch_idx == 1 and ep > 1:
                tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                warnings.simplefilter(action='ignore', category=FutureWarning)
                # X_tsne = tsne.fit_transform(output[:batch_size, :].cpu().detach().numpy())
                # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
                # plt.figure(figsize=(4, 4))
                # for i in range(X_norm.shape[0]):
                #     if gt_s[i] == 0:
                #         plt.scatter(X_norm[i, 0], X_norm[i, 1], s=30, marker='s', color='k')
                #     else:
                #         plt.scatter(X_norm[i, 0], X_norm[i, 1], s=30, marker='o', color='k')
                # plt.xticks([])
                # plt.yticks([])
                save_name = (args.save + '/target_summary')
                save_source_name = (args.save + '/source_summary')
                if os.path.exists(save_name) == False:
                    os.mkdir(save_name)
                if os.path.exists(save_source_name) == False:
                    os.mkdir(save_source_name)
                value, y_pre_1, y_pre_2, y_true, output = test_totalmix(ep, target_test_totalmix_loader, G2, G3, G4, F1, F2, writer, save_name, args)
                value_source, y_pre_source_1, y_pre_source_2, y_source_true, source_output = test_totalmix(ep, source_test_totalmix_loader,
                                                                        G2, G3, G4, F1, F2, writer, save_source_name, args)
                if ep == 1:
                    value_max = value
                else:
                    if value > value_max:
                        value_max = value
                        if os.path.exists(args.save + '/sub_target') == False:
                            os.mkdir(args.save + '/sub_target')
                        if os.path.exists(args.save + '/sub_source') == False:
                            os.mkdir(args.save + '/sub_source')
                        # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                        # warnings.simplefilter(action='ignore', category=FutureWarning)
                        # X_tsne = tsne.fit_transform(output.cpu().detach().numpy())
                        # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                        # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
                        # for i in range(X_norm.shape[0]):
                        #     if gt_t[i] == 0:
                        #         plt.scatter(X_norm[i, 0], X_norm[i, 1], s=30, marker='s', color='cornflowerblue')
                        #     else:
                        #         plt.scatter(X_norm[i, 0], X_norm[i, 1], s=30, marker='o', color='cornflowerblue')
                        # plt.xticks([])
                        # plt.yticks([])
                        # plt.savefig(args.save + '/sub_target/minist_' + str(ep) + '.jpg')
                        # plt.show()
                        # plt.close('all')

                        scio.savemat(
                            os.path.join(args.save + '/sub_target/predict.mat'),
                            {'epoch': ep, 'Y_true': y_true, 'Y_pre_1': y_pre_1, 'Y_pre_2': y_pre_2})
                        scio.savemat(
                            os.path.join(args.save + '/sub_source/predict.mat'),
                            {'epoch': ep, 'Y_true': y_source_true, 'Y_pre_1': y_pre_source_1, 'Y_pre_2': y_pre_source_2})
                writer.add_scalar('accuracy', value_max, global_step=ep)
                G2.train()
                G3.train()
                G4.train()
                F1.train()
                F2.train()


def test_totalmix(epoch, target_test_totalmix_loader, G2, G3, G4, F1, F2, writer, save_name, args):
    G2.eval()
    G3.eval()
    G4.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    size = 0
    y_pre_1 = []
    y_pre_2 = []
    y_true = []

    for batch_idx, (erp_data, ECG_data, EMG_data, gt_t) in enumerate(target_test_totalmix_loader):
        y_true.append(gt_t.cpu().numpy().squeeze())
        if args.cuda:
            erp_data, gt_t = erp_data.cuda(), gt_t.cuda()
            ECG_data, gt_t = ECG_data.cuda(), gt_t.cuda()
            EMG_data, gt_t = EMG_data.cuda(), gt_t.cuda()
        with torch.no_grad():
            data_erp, target1 = Variable(erp_data), Variable(gt_t)
            data_ECG, target1 = Variable(ECG_data), Variable(gt_t)
            data_EMG, target1 = Variable(EMG_data), Variable(gt_t)
        output_erp = G2(data_erp)
        output_ECG = G3(data_ECG)
        output_EMG = G4(data_EMG)
        output = torch.stack((output_erp, output_ECG, output_EMG), 1)
        output1, flatten_data = F1(output)
        output2, flatten_data = F2(output)
        # output1 = F.softmax(output1, dim=1)
        # output2 = F.softmax(output2, dim=1)
        test_loss += F.nll_loss(output1, target1.long().squeeze()).item()
        pred = output1.data.max(1)[1]  # get the index of the max log-probability
        y_pre_1.append(pred.cpu().numpy().squeeze())
        correct1 += pred.eq(target1.data.squeeze().long()).cpu().sum()
        pred = output2.data.max(1)[1]  # get the index of the max log-probability
        y_pre_2.append(pred.cpu().numpy().squeeze())
        k = target1.data.size()[0]
        correct2 += pred.eq(target1.data.squeeze().long()).cpu().sum()
        correct = max([correct1, correct2])

        size += k

    test_loss = test_loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)\n'.format(
        test_loss, correct, size,
        100. * correct1 / size, 100. * correct2 / size))
    writer.add_scalar('loss_test', test_loss, global_step=epoch)
    # if 100. * correct / size > 67 or 100. * correct2 / size > 67:
    value = max(100. * correct1 / size, 100. * correct2 / size)

    if value > 58:
        torch.save(F1.state_dict(), save_name + '/' + '_' + str(value) + '_' + 'F1.pth')
        torch.save(F2.state_dict(), save_name + '/' + '_' + str(value) + '_' + 'F2.pth')
        torch.save(G2.state_dict(), save_name + '/' + '_' + str(value) + '_' + 'G2.pth')
        torch.save(G3.state_dict(), save_name + '/' + '_' + str(value) + '_' + 'G3.pth')
        torch.save(G4.state_dict(), save_name + '/' + '_' + str(value) + '_' + 'G4.pth')
    return value, y_pre_1, y_pre_2, y_true, output

