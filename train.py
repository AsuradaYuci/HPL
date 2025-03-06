import copy
import time
import datetime
import logging
import torch
from tools.utils import AverageMeter
from visualize import reverse_normalize, visual_batch
import random
import numpy as np
from losses.triplet_loss import TripletLoss_euc
# from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
# from tools.faiss_rerank import compute_jaccard_distance
from losses.contrastive_loss import ContrastiveLoss2

contrast = ContrastiveLoss2().cuda()
triplet_loss_2 = TripletLoss_euc(margin=0.3, normalize_feature=True).cuda()
triplet_loss_22 = TripletLoss_euc(margin=0.3, normalize_feature=True).cuda()


def mseloss(feat1, feat2):  # [64, 2048], [64, 2048]

	dist = torch.pow(torch.abs(feat1 - feat2), 2).sum(dim=-1)
	# loss = (1. / (1. + torch.exp(-dist))).mean()

	loss = dist.mean()

	return loss


# cluster = MiniBatchKMeans(n_clusters=2, init_size=2, batch_size=8, max_iter=100)
# cluster = KMeans(n_clusters=3, random_state=10, max_iter=20)
# cluster2 = DBSCAN(eps=0.5, min_samples=2, metric='precomputed', n_jobs=-1)


def get_raw_images(images):
	b, c, h, w = images.size()
	image = images.view(b, c, h, w)
	imgs_vis = []
	for k in range(b):
		img = image[k].unsqueeze(0)  # torch.Size([1, 3, 256, 128])
		img = reverse_normalize(img)
		imgs_vis.append(img)
	return imgs_vis


def keshihua(cam, imgs, i, savedir):
	visual_batch(cam, imgs, i, savedir)
	return print('done')


def train_cal(config, epoch, model, classifier, criterion_cla, criterion_pair, optimizer,  trainloader, ):
	logger = logging.getLogger('reid.train')
	batch_cla_loss = AverageMeter()
	batch_pair_loss1 = AverageMeter()
	batch_pair_loss2 = AverageMeter()
	batch_pair_loss3 = AverageMeter()
	batch_clo_loss = AverageMeter()
	batch_adv_loss = AverageMeter()
	batch_MSE_loss1 = AverageMeter()
	batch_MSE_loss2 = AverageMeter()
	corrects = AverageMeter()
	clothes_corrects = AverageMeter()
	batch_time = AverageMeter()
	data_time = AverageMeter()

	model.train()
	classifier.train()
	# clothes_classifier.train()

	end = time.time()
	for batch_idx, (imgs, mask, pids, camids, clothes_ids, _) in enumerate(trainloader):
		# Get all positive clothes classes (belonging to the same identity) for each sample
		# flip_imgs = torch.flip(imgs, [3])  # torch.Size([128, 3, 256, 128])
		# raw_imgs = get_raw_images(imgs)
		# raw_imgs_flip = get_raw_images(flip_imgs)
		# keshihua(raw_imgs_flip, raw_imgs, batch_idx, 'epoch{}soft_mask_total_1epoch'.format(batch_idx))

		b, c, h, w = imgs.shape
		# pos_mask = pid2clothes[pids]
		pids_single = pids[0:len(pids):8]
		pids_single_c = torch.cat((pids_single, pids_single), dim=0)
		assert len(pids_single_c) == 8

		imgs, pids = imgs.cuda(), pids.cuda(),
		# Measure data loading time
		data_time.update(time.time() - end)

		mask_i2 = mask.squeeze().argmax(dim=1).unsqueeze(dim=1)  # [64, 1, 256, 128] 对mask中的值进行排序。得到解析结果
		# mask_i = mask_i2.expand_as(imgs)  # [64, 3, 256, 128]  for ltcc
		# mask = mask.cuda()
		# mask_i2 = mask
		mask_expand = mask_i2.expand_as(imgs)
		imgs_a = copy.deepcopy(imgs)
		# if random.uniform(0, 1) >= 0.5:
		# upper clothes mask
		index1 = np.random.permutation(b)
		img_r = imgs[index1]
		mask_r = mask_expand[index1]
		imgs_a[mask_expand == 2] = img_r[mask_r == 2]
		# else:
		# down clothes mask
		index2 = np.random.permutation(b)
		img_r = imgs[index2]
		mask_r = mask_expand[index2]
		imgs_a[mask_expand == 3] = img_r[mask_r == 3]

		imgs_c = torch.cat([imgs, imgs_a], dim=0)  # torch.Size([128, 3, 384, 192])
		features, features_bn = model(imgs_c)  # [2b, 2048]  torch.Size([128, 4096])
		# features [2b,4096]
		f1, f2 = torch.split(features, [b, b], dim=0)  # torch.Size([32, 4096])
		# dist_euc = get_intra_clothes(features, b)
		# ###################   1. CPL
		dist_euc1, clothes_centers = get_intra_clothes(f1, b)  # f1是没有数据增强的样本
		dist_euc2, clothes_centers2 = get_intra_clothes(f2, b)  # f1是没有数据增强的样本
		# # ################## 2.衣服无关，pid相关 PID的平均值a  PPL
		features_split = f1.view(b // 8, 8, -1)  # 4,8,4096
		features_mean = torch.mean(features_split, dim=1)  # 4, 4096
		features_mean_proxy = features_mean.unsqueeze(1).expand_as(clothes_centers).reshape(-1, 4096)
		pids_cont = pids_single.unsqueeze(1).expand((4, 2)).flatten()
		constrast_loss = contrast(clothes_centers.reshape(-1, 4096), features_mean_proxy, pids_cont)
		# constrast_loss = contrast(f1, features_mean_proxy, pids)

		features_split2 = f2.view(b // 8, 8, -1)  # 4,8,4096
		features_mean2 = torch.mean(features_split2, dim=1)  # 4, 4096
		# 1.ID cls
		outputs = classifier(features_bn)  # torch.Size([128, 77])
		_, preds = torch.max(outputs.data, 1)
		features_clothes = features_bn[0:b]  # torch.Size([64, 4096])
		# 2.Clothes-ID cls
		# pred_clothes = clothes_classifier(features_clothes.detach())  # torch.Size([64, 256])

		# Update the clothes discriminator
		# 3. Clothes_loss
		# clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
		# if epoch <= 25:  # 25
		# 	optimizer_cc.zero_grad()
		# 	clothes_loss.backward()
		# 	optimizer_cc.step()

		# Update the backbone
		# # 22.Clothes-ID cls
		# new_pred_clothes = clothes_classifier(features_clothes)
		# _, clothes_preds = torch.max(new_pred_clothes.data, 1)

		# Compute loss
		# 4. CE los
		pids_c = torch.cat([pids, pids], dim=0)
		cla_loss = criterion_cla(outputs, pids_c)
		# mse loss
		# mse_loss = mseloss(features[0:b], features[b:])
		# 5. triplet loss
		# pair_loss = criterion_pair(features_clothes, pids)  # triplet
		features_mean_concat = torch.cat((features_mean, features_mean2), dim=0)  # torch.Size([8, 4096])
		# pids_single_c  PPL
		pair_loss1 = triplet_loss_2(features_mean, pids_single)
		pair_loss2 = triplet_loss_22(features_mean2, pids_single)
		pair_loss = triplet_loss_22(features_mean_concat, pids_single_c)
		# 6. CA loss
		# adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
		if epoch >= config.TRAIN.START_EPOCH_ADV:
			loss = cla_loss + (dist_euc1 + dist_euc2 + pair_loss + constrast_loss + pair_loss1 + pair_loss2) * 0.1
		else:
			loss = cla_loss

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# statistics
		corrects.update(torch.sum(preds == pids_c.data).float() / pids_c.size(0), pids_c.size(0))
		# clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float() / clothes_ids.size(0),
		#                         clothes_ids.size(0))
		batch_cla_loss.update(cla_loss.item(), pids.size(0))
		batch_pair_loss1.update(constrast_loss.item(), pids.size(0))
		# batch_pair_loss2.update(pair_loss2.item(), pids.size(0))
		batch_pair_loss3.update(pair_loss.item(), pids.size(0))
		# batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
		# batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
		batch_MSE_loss1.update(dist_euc1.item(), pids.size(0))
		batch_MSE_loss2.update(dist_euc2.item(), pids.size(0))
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if (batch_idx + 1) % 50 ==0:
			logger.info('Epoch[{}] Iteration[{}/{}]'
			            'Time:{batch_time.sum:.1f}s '
			            'Data:{data_time.sum:.1f}s '
			            'ClaLoss:{cla_loss.avg:.4f} '
			            'contrast1:{pair_loss1.avg:.4f} '
			            'PairLoss3:{pair_loss3.avg:.4f} '
			            'mseLoss1:{mse_loss1.avg:.4f} '
			            'mseLoss2:{mse_loss2.avg:.4f} '
			            'Acc:{acc.avg:.2%} '.format(
				epoch + 1, batch_idx+1, len(trainloader),
				batch_time=batch_time, data_time=data_time,
				cla_loss=batch_cla_loss,
				pair_loss1=batch_pair_loss1,
				pair_loss3=batch_pair_loss3,
				mse_loss1=batch_MSE_loss1,
				mse_loss2=batch_MSE_loss2,
				# clo_loss=batch_clo_loss,
				# adv_loss=batch_adv_loss,
				acc=corrects,
				# clo_acc=clothes_corrects
			))


def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(x, y.t(), beta=1, alpha=-2)
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist


def get_intra_clothes(features_before_bn, b):  #
	# B = b//8
	# features_parpear_clothes_cluter = features_before_bn.view(B, 8, -1)  # 4,
	# cluster_list = []
	# for j in range(B):
		# rerank_dist2 = compute_jaccard_distance(features_parpear_clothes_cluter[j].detach().cpu(), k1=20,
		#                                         k2=6)
		# feat_j = features_parpear_clothes_cluter[j].detach().cpu().numpy()  # torch.Size([8, 4096])
		# pseudo_labels = cluster.fit(feat_j)
		# pseudo_labels = cluster.fit_predict(feat_j)

		# pseudo_labels2 = cluster2.fit_predict(rerank_dist2)
		# cluster_centers = pseudo_labels
	features_parpear_clothes_center = features_before_bn.view(b // 8, 2, 4, -1)  # 4, 2, 4, 4096
	features_clothes_centers = torch.mean(features_parpear_clothes_center, dim=2)  # 4, 2, 4096
	f_1 = features_clothes_centers[:, 0, :]  # 4, 4096
	f_2 = features_clothes_centers[:, 1, :]  # 4, 4096
	# features_parpear_clothes_center2 = features_parpear_clothes_center.view(b//8, 2, 4, -1)
	f = torch.stack((f_2, f_1), dim=1).unsqueeze(dim=-2)  # [4, 2, 4096]=>[4, 2, 1, 4096]交换一下,准备
	f_expand = f.expand_as(features_parpear_clothes_center).reshape_as(features_before_bn).contiguous()  # 32, 4096
	loss_clothes_aware = 0
	for index_c in range(b):
		f_index_c = features_before_bn[index_c].unsqueeze(0)
		f_expand_index_c = f_expand[index_c].unsqueeze(0)
		dist_index_c = euclidean_dist(f_index_c, f_expand_index_c)
		loss_clothes_aware += dist_index_c
	loss_clothes_aware = loss_clothes_aware / b
	return loss_clothes_aware, features_clothes_centers