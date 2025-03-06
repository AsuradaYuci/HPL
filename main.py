import os
import sys
import time
import datetime
import argparse
import logging
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import distributed as dist


from configs.default_img import get_img_config
from configs.default_vid import get_vid_config
from data import build_dataloader
from models import build_model
from losses import build_losses
from tools.utils import save_checkpoint, set_seed, get_logger
from train import train_cal
from test import test, test_prcc, extect_prcc
from tools.faiss_rerank import compute_jaccard_distance
from sklearn.cluster import DBSCAN
from collections import defaultdict

import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT
import data.transformers_double as T2
from torch.utils.data import DataLoader
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, VideoDataset, ImageDatasetGcnMask, ImageDataset_unsuper

from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.last import LaST
from data.datasets.ccvid import CCVID
from data.datasets.deepchange import DeepChange
from data.datasets.vcclothes import VCClothes, VCClothesSameClothes, VCClothesClothesChanging


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'vcclothes': VCClothes,
    'vcclothes_sc': VCClothesSameClothes,
    'vcclothes_cc': VCClothesClothesChanging,
    'last': LaST,
    'ccvid': CCVID,
    'deepchange': DeepChange,
}

def get_names():
    return list(__factory.keys())


def build_dataset(config):
    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)
    return dataset


def build_img_transforms_trian(config):
    transform_train = T2.Compose([
        T2.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T2.RandomCrop((config.DATA.HEIGHT, config.DATA.WIDTH), p=config.AUG.RC_PROB),
        T2.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T2.ToTensor(),
        T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T2.RandomErasing(probability=config.AUG.RE_PROB)
        #T2.RandomErase(probability=config.AUG.RE_PROB)
    ])

    return transform_train


def build_img_transforms_test(config):
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_test



VID_DATASET = ['ccvid']

# Draw Curve
#-----------
x_epoch = []
y_acc = {} # loss history
y_acc['map'] = []
y_acc['rank1'] = []

fig = plt.figure()
ax0 = fig.add_subplot(111, title="eval performance")


def draw_curve(config, current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_acc['map'], 'bo-', label='map')
    ax0.plot(x_epoch, y_acc['rank1'], 'ro-', label='rank1')

    if current_epoch == 0:
        ax0.legend()
    fig.savefig(os.path.join(config.OUTPUT, 'eval_performance.jpg'))


def cal_dist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def generate_pseudo_labels1(cluster_id, inputFeat):
    with_id = inputFeat[cluster_id != -1]  # torch.Size([3330, 2048])
    witho_id = inputFeat[cluster_id == -1]  # torch.Size([9606, 2048])
    disMat = cal_dist(with_id, witho_id)  # torch.Size([3330, 9606])
    # relabel images
    neighbour = disMat.argmin(0).cpu().numpy()  # <class 'tuple'>: (9606,)
    newID = cluster_id[cluster_id != -1][neighbour]  # <class 'tuple'>: (9606,)
    cluster_id[cluster_id == -1] = newID
    # return torch.from_numpy(cluster_id)
    return cluster_id


def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='prcc', help="ltcc, prcc, vcclothes, ccvid, last, deepchange")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--amp', action='store_true', help="automatic mixed precision")
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    if args.dataset in VID_DATASET:
        config = get_vid_config(args)
    else:
        config = get_img_config(args)

    return config


def main(config):
    # Build dataloader
    dataset = build_dataset(config)  # 1.数据集信息
    # image dataset
    transform_train = build_img_transforms_trian(config)  # 2.数据增强
    transform_test = build_img_transforms_test(config)  # 2.数据增强

    galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                                sampler=DistributedInferenceSampler(dataset.gallery),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=False, shuffle=False)

    if config.DATA.DATASET == 'prcc':
        queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test),
                                       sampler=DistributedInferenceSampler(dataset.query_same),
                                       batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=False, shuffle=False)
        queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test),
                                       sampler=DistributedInferenceSampler(dataset.query_diff),
                                       batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                       pin_memory=True, drop_last=False, shuffle=False)

    else:
        queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test),
                                  sampler=DistributedInferenceSampler(dataset.query),
                                  batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                  pin_memory=True, drop_last=False, shuffle=False)

    # pid2clothes = torch.from_numpy(dataset.pid2clothes)

    # Build model
    model, classifier = build_model(config, dataset.num_train_pids)
    # Build identity classification loss, pairwise loss, clothes classificaiton loss, and adversarial loss.
    criterion_cla, criterion_pair = build_losses(config)
    # Build optimizer
    parameters = list(model.parameters()) + list(classifier.parameters())
    if config.TRAIN.OPTIMIZER.NAME == 'adam':  # 111
        optimizer = optim.Adam(parameters, lr=config.TRAIN.OPTIMIZER.LR, 
                               weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        # optimizer_cc = optim.Adam(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR,
        #                           weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
    elif config.TRAIN.OPTIMIZER.NAME == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9, 
                              weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
        # optimizer_cc = optim.SGD(clothes_classifier.parameters(), lr=config.TRAIN.OPTIMIZER.LR, momentum=0.9,
        #                       weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY, nesterov=True)
    else:
        raise KeyError("Unknown optimizer: {}".format(config.TRAIN.OPTIMIZER.NAME))
    # Build lr_scheduler
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.LR_SCHEDULER.STEPSIZE, 
                                         gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE)

    start_epoch = config.TRAIN.START_EPOCH  # 0
    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME)
        model.load_state_dict(checkpoint['model_state_dict'])
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        # if config.LOSS.CAL == 'calwithmemory':
        #     criterion_adv.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        # else:
        #     clothes_classifier.load_state_dict(checkpoint['clothes_classifier_state_dict'])
        start_epoch = checkpoint['epoch']

    local_rank = dist.get_rank()  # 0
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda(local_rank)
    model = model.cuda(local_rank)
    classifier = classifier.cuda(local_rank)

    # clothes_classifier = clothes_classifier.cuda(local_rank)
    torch.cuda.set_device(local_rank)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        best_path = '/media/ycy/18b21f78-77a1-403a-959e-d65e937da92b/Simple-CCReID-main/logs/prcc/res50-cels-cal_256*128_avgpool_noflip_test/checkpoint_ep45.pth.tar'
        # logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(best_path)
        checkpoint_model = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint_model)
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test(config, model, queryloader, galleryloader, dataset)
        return
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], output_device=local_rank)
    # if config.LOSS.CAL != 'calwithmemory':
    #     clothes_classifier = nn.parallel.DistributedDataParallel(clothes_classifier,
    #                                                              device_ids=[local_rank],
    #                                                              output_device=local_rank)

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    logger.info("==> Start training")
    eps = 0.6  # 0.4
    print('Clustering criterion: eps: {:.3f}'.format(eps))
    cluster = DBSCAN(eps=eps, min_samples=8, metric='precomputed', n_jobs=-1)
    for epoch in range(start_epoch, config.TRAIN.MAX_EPOCH):

        trainloader2 = DataLoaderX(dataset=ImageDataset_unsuper(dataset.train, transform=transform_test),
                                   sampler=DistributedInferenceSampler(dataset.train),
                                   batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                   pin_memory=True, drop_last=False, shuffle=False)

        with torch.no_grad():
            pid2feat_dic = defaultdict(list)
            features, pids_train = extect_prcc(model, trainloader2, dataset)

            for index_i, pid_i in enumerate(pids_train):
                pid2feat_dic[pid_i].append(torch.from_numpy(features[index_i]))
            pids_list = list(pid2feat_dic.keys())
            pseudo_labels_list = []
            num_init = 0
            for pids_index in pids_list:
                features_i = torch.stack(pid2feat_dic[pids_index], 0)

                rerank_dist_i = compute_jaccard_distance(features_i, k1=20,
                                                       k2=6)  # k1=30 k2=6  <class 'tuple'>: (12936, 12936)
                pseudo_labels = cluster.fit_predict(rerank_dist_i)
                if -1 in pseudo_labels:
                    pseudo_labels = generate_pseudo_labels1(pseudo_labels, features_i)  # torch.Size([16522])

                pseudo_labels = pseudo_labels + num_init + 1
                num_init = pseudo_labels.max()
                pseudo_labels_list.extend(pseudo_labels)

        # 更新伪标签数据集
        pseudo_labeled_dataset = []  # 17896
        for i, ((img_dir, pidsj, camid, _, msk_path, path_flag), clothes_label) in enumerate(zip(dataset.train, pseudo_labels_list)):
            # if label != -1:
            pseudo_labeled_dataset.append((img_dir, pidsj, camid, clothes_label, msk_path, path_flag))

        # 更新训练的dataloarder
        train_sampler = DistributedRandomIdentitySampler(pseudo_labeled_dataset,
                                                         num_instances=config.DATA.NUM_INSTANCES,
                                                         seed=config.SEED)  # 训练采样
        trainloader3 = DataLoaderX(dataset=ImageDatasetGcnMask(pseudo_labeled_dataset, transform=transform_train),
                                  sampler=train_sampler,
                                  batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                  pin_memory=True, drop_last=True)  # 训练集封装

        train_sampler.set_epoch(epoch)
        start_train_time = time.time()

        train_cal(config, epoch, model, classifier, criterion_cla, criterion_pair, optimizer, trainloader3)
        train_time += round(time.time() - start_train_time)        
        
        if (epoch+1) > config.TEST.START_EVAL and config.TEST.EVAL_STEP > 0 and \
            (epoch+1) % config.TEST.EVAL_STEP == 0 or (epoch+1) == config.TRAIN.MAX_EPOCH:
            logger.info("==> Test")
            torch.cuda.empty_cache()
            if config.DATA.DATASET == 'prcc':
                rank1, map = test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                rank1, map = test(config, model, queryloader, galleryloader, dataset)
            y_acc['rank1'].append(rank1)
            y_acc['map'].append(map)
            # draw curve
            draw_curve(config, epoch)

            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            model_state_dict = model.module.state_dict()
            classifier_state_dict = classifier.module.state_dict()

            # clothes_classifier_state_dict = clothes_classifier.module.state_dict()
            if local_rank == 0:
                save_checkpoint({
                    'model_state_dict': model_state_dict,
                    'classifier_state_dict': classifier_state_dict,
                    # 'clothes_classifier_state_dict': clothes_classifier_state_dict,
                    'rank1': rank1,
                    'epoch': epoch,
                }, is_best, osp.join(config.OUTPUT, 'checkpoint_ep.pth.tar'))
        scheduler.step()

    logger.info("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    logger.info("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    

if __name__ == '__main__':
    config = parse_option()
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU  # 0,1
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    set_seed(config.SEED + local_rank)
    # get logger
    run = 0
    if not config.EVAL_MODE:
        while osp.exists("%s" % (osp.join(config.OUTPUT, 'log_train{}.txt'.format(run)))):
            run += 1
        output_file = osp.join(config.OUTPUT, 'log_train{}.txt'.format(run))

    else:
        while osp.exists("%s" % (osp.join(config.OUTPUT, 'log_test{}.txt'.format(run)))):
            run += 1
        output_file = osp.join(config.OUTPUT, 'log_test{}.txt'.format(run))

    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    main(config)
