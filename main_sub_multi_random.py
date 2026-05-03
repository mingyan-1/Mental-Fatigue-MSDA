# -*- coding:utf-8 -*-
"""
作者：
日期：2022年10月13日
"""
from __future__ import print_function

import data_load_multisub
from train_sub_totamix import *
from train_notransfer import *
from tensorboardX import SummaryWriter
import numpy as np
import scipy.io as scio
import random
import os
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def normalize_dataset(dataset, mean_std_ecg, mean_std_meg, mean_std_megfeature):
    normalized_data = []
    for i in range(len(dataset)):
        ecg, feature, meg, meg_feature, label, sub = dataset[i]  # 忽略 meg
        ecg = (ecg - mean_std_ecg["mean"]) / (mean_std_ecg["std"] + 1e-8)
        meg = (meg - mean_std_meg["mean"]) / (mean_std_meg["std"] + 1e-8)
        meg_feature = (meg_feature - mean_std_megfeature["mean"]) / (mean_std_megfeature["std"] + 1e-8)
        ecg = ecg.float()
        feature = feature.float()
        meg = meg.float()
        meg_feature = meg_feature.float()
        normalized_data.append((ecg, feature, meg, meg_feature, label, sub))
    return normalized_data

for repeat in range(5, 6):
    for sub in range(1, 32):
        for source_num in range(2, 10):
            # Training settings
            G2 = G_erp()
            G3 = G_ECG()
            G4 = G_EMG()
            F1 = ResClassifier(3, 2)
            F2 = ResClassifier(3, 2)
            parser = argparse.ArgumentParser(description='Visda Classification')
            parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                                help='input batch size for training (default: 64)')
            parser.add_argument('--epochs', type=int, default=50+(source_num//2)*25, metavar='N',
                                help='number of epochs to train (default: 10)')
            parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                                help='learning rate (default: 0.001)')
            parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                                help='SGD momentum (default: 0.5)')
            parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                                help='the name of optimizer')
            parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='disables CUDA training')
            parser.add_argument('--seed', type=int, default=1, metavar='S',
                                help='random seed (default: 1)')
            parser.add_argument('--log-interval', type=int, default=8, metavar='N',
                                help='how many batches to wait before logging training status')
            parser.add_argument('--num_k', type=int, default=4, metavar='K',
                                help='how many steps to repeat the generator update')
            parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                                help='how many layers for classifier')
            parser.add_argument('--name', type=str, default='board', metavar='B',
                                help='board dir')
            parser.add_argument('--save', type=str, default=('../visda_classification/save_submulititotal_momentum_random3/save_noran_'+str(repeat)+'/sub'+str(sub)+'/source_num'+str(source_num)), metavar='B',
                                help='board dir')
            parser.add_argument('--root_path', type=str, default='../visda_classification/data_new/data_sub_zscore', metavar='B',
                                help='directory of source datasets')
            args = parser.parse_args()
            args.cuda = not args.no_cuda and torch.cuda.is_available()
            root_path = args.root_path
            num_k = args.num_k
            num_layer = args.num_layer
            batch_size = args.batch_size
            writer = SummaryWriter(args.save + '/summary/snap'+str(sub)+'/')
            if os.path.exists(args.save + '/summary/') == False:
                os.mkdir(args.save + '/summary/')

            img_transform_source = transforms.Compose([
                transforms.ToTensor()
            ])

            img_transform_target = transforms.Compose([
                transforms.ToTensor()
            ])

            # data_transforms = {
            #     train_path: transforms.Compose([
            #         transforms.ToTensor()
            #     ]),
            #     val_path: transforms.Compose([
            #         transforms.ToTensor()
            #     ])
            # }
            # file_path = './coral_31.mat'
            # with open(file_path, 'rb') as f:
            #     coral = scio.loadmat(f)
            #     coral = coral['coral']
            # target_coral = coral[sub-1, :]
            # b = sorted(enumerate(target_coral), key=lambda target_coral: target_coral[1])  # x[1]是因为在enumerate(a)中，a数值在第1位
            # index = [target_coral[0] for target_coral in b]  # 获取排序好后b坐标,下标在第0位
            index = list(set(list(range(0, 31))) - {sub-1})
            random.shuffle(index)
            select_source = index[1:source_num+1]
            kwargs = {'num_workers': 4, 'pin_memory': True}
            target_file = scio.loadmat(root_path+'/sub_'+str(sub)+'/train.mat')
            target_data = np.repeat(target_file['data'], repeats=source_num, axis=0)
            target_gt = np.repeat(target_file['gt'], repeats=source_num, axis=1)
            source_data = np.zeros((192, 2416, source_num))
            source_gt = np.zeros((192, source_num))
            for source_repeat in range(0, source_num):
                source_file = scio.loadmat(root_path+'/sub_'+str(select_source[source_repeat]+1)+'/train.mat')
                source_data[:, :, source_repeat] = source_file['data']
                source_gt[:, source_repeat] = source_file['gt'].T
            mean_std = {
                "mean": np.median(source_data, axis=0),
                "std": np.percentile(source_data, 75, axis=0)-np.percentile(source_data, 25, axis=0)
            }
            source_data = (source_data - mean_std["mean"]) / (mean_std["std"] + 1e-8)
            mean_std_target = {
                "mean": np.mean(mean_std["mean"], axis=1),
                "std": np.mean(mean_std["std"], axis=1)
            }
                # if source_repeat == 0:
                #     source_data = source_file['data']
                #     source_gt = source_file['gt']
                # else:
                #     source_data = np.concatenate((source_data, source_file['data']), axis=0)
                #     source_gt = np.concatenate((source_gt, source_file['gt']), axis=1)
            target_test_file = scio.loadmat(root_path + '/sub_' + str(sub) + '/total.mat')
            target_test_data = target_test_file['data']
            target_test_data = (target_test_data - mean_std_target["mean"]) / (mean_std_target["std"] + 1e-8)
            target_test_gt = target_test_file['gt']
            for source_repeat in range(0, source_num):
                source_file = scio.loadmat(root_path + '/sub_' + str(select_source[source_repeat]+1) + '/test.mat')
                if source_repeat == 0:
                    source_test_data = source_file['data']                    
                    source_test_gt = source_file['gt']
                else:
                    source_test_data = np.stack((source_test_data, source_file['data']), axis=2)
                    source_test_gt = np.concatenate((source_test_gt, source_file['gt']), axis=0)
            if source_repeat == 0:
                source_test_data = (source_test_data - mean_std_target["mean"]) / (mean_std_target["std"] + 1e-8)
            else: 
                source_test_data = (source_test_data - mean_std["mean"]) / (mean_std["std"] + 1e-8)
                reshaped = source_test_data.transpose(0, 2, 1)
                source_test_data = reshaped.reshape(-1, 2416)
            source_totalmix_loader = data_load_multisub.load_totalmix_training(target_data, source_data, target_gt, source_gt, batch_size, kwargs)
            target_test_totalmix_loader = data_load_multisub.load_totalmix_testing(target_test_data, target_test_gt, batch_size, kwargs)
            source_test_totalmix_loader = data_load_multisub.load_totalmix_testing(source_test_data, source_test_gt, batch_size, kwargs)

            if args.cuda:
                use_gpu = torch.cuda.is_available()
            torch.manual_seed(args.seed)
            if args.cuda:
                torch.cuda.manual_seed(args.seed)
            F1.apply(weights_init)
            F2.apply(weights_init)
            lr = args.lr

            if args.cuda:
                G2.cuda()
                G3.cuda()
                G4.cuda()
                F1.cuda()
                F2.cuda()
            if args.optimizer == 'momentum':
                optimizer_g2 = optim.SGD(list(G2.parameters()), lr=args.lr, weight_decay=0.0005)
                optimizer_g3 = optim.SGD(list(G3.parameters()), lr=args.lr, weight_decay=0.0005)
                optimizer_g4 = optim.SGD(list(G4.parameters()), lr=args.lr, weight_decay=0.0005)
                optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                                        weight_decay=0.0005)
            elif args.optimizer == 'adam':
                optimizer_g2 = optim.Adam(G2.parameters(), lr=args.lr, weight_decay=0.0005)
                optimizer_g3 = optim.Adam(G3.parameters(), lr=args.lr, weight_decay=0.0005)
                optimizer_g4 = optim.Adam(G4.parameters(), lr=args.lr, weight_decay=0.0005)
                optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
            else:
                optimizer_g2 = optim.Adadelta(G2.parameters(), lr=args.lr, weight_decay=0.0005)
                optimizer_g3 = optim.Adadelta(G3.parameters(), lr=args.lr, weight_decay=0.0005)
                optimizer_g4 = optim.Adadelta(G4.parameters(), lr=args.lr, weight_decay=0.0005)
                optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
            train_totalmix(args.epochs + 1, source_totalmix_loader, target_test_totalmix_loader, source_test_totalmix_loader, G2, G3, G4, F1, F2, optimizer_g2, optimizer_g3, optimizer_g4, optimizer_f, batch_size, num_k, writer, sub, args)
            # train_notransfer(args.epochs + 1, source_totalmix_loader, target_test_totalmix_loader,source_test_totalmix_loader, G2, G3, G4, F1, optimizer_g2, optimizer_g3, optimizer_g4, optimizer_f, batch_size, writer, args)


