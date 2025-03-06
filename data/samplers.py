#!/usr/bin/env Python
# coding=utf-8
import copy
import math
import random
import numpy as np
from torch import distributed as dist
from collections import defaultdict
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
	"""
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

	def __init__(self, data_source, num_instances=4):
		self.data_source = data_source
		self.num_instances = num_instances
		self.index_dic = defaultdict(list)
		for index, (_, pid, _, _) in enumerate(data_source):
			self.index_dic[pid].append(index)
		self.pids = list(self.index_dic.keys())
		self.num_identities = len(self.pids)

		# compute number of examples in an epoch
		self.length = 0
		for pid in self.pids:
			idxs = self.index_dic[pid]
			num = len(idxs)
			if num < self.num_instances:
				num = self.num_instances
			self.length += num - num % self.num_instances

	def __iter__(self):
		list_container = []

		for pid in self.pids:
			idxs = copy.deepcopy(self.index_dic[pid])
			if len(idxs) < self.num_instances:
				idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
			random.shuffle(idxs)
			batch_idxs = []
			for idx in idxs:
				batch_idxs.append(idx)
				if len(batch_idxs) == self.num_instances:
					list_container.append(batch_idxs)
					batch_idxs = []

		random.shuffle(list_container)

		ret = []
		for batch_idxs in list_container:
			ret.extend(batch_idxs)

		return iter(ret)

	def __len__(self):
		return self.length


class DistributedRandomIdentitySampler(Sampler):
	"""
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
    - data_source (Dataset): dataset to sample from.
    - num_instances (int): number of instances per identity.
    - num_replicas (int, optional): Number of processes participating in
        distributed training. By default, :attr:`world_size` is retrieved from the
        current distributed group.
    - rank (int, optional): Rank of the current process within :attr:`num_replicas`.
        By default, :attr:`rank` is retrieved from the current distributed group.
    - seed (int, optional): random seed used to shuffle the sampler.
        This number should be identical across all
        processes in the distributed group. Default: ``0``.
    """

	def __init__(self, data_source, num_instances=4,
	             num_replicas=None, rank=None, seed=0):
		if num_replicas is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			num_replicas = dist.get_world_size()  # 2
		if rank is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			rank = dist.get_rank()  # 0
		if rank >= num_replicas or rank < 0:
			raise ValueError(
				"Invalid rank {}, rank should be in the interval"
				" [0, {}]".format(rank, num_replicas - 1))
		self.num_replicas = num_replicas
		self.rank = rank
		self.seed = seed
		self.epoch = 0

		self.data_source = data_source
		self.num_instances = num_instances
		self.index_dic = defaultdict(list)
		self.index_pid = defaultdict(list)  # 索引对应的pid
		self.clothes_index = defaultdict(list)  # 索引对应的衣服id
		self.pid_clothes = defaultdict(set)  # 一个ID有几件衣服
		self.pid_index = defaultdict(list)
		# 构建一个python字典嵌套字典构造

		self.pid_clothes_index = dict()

		for index, (_, pid, _, clothes_id, _, _) in enumerate(data_source):  # change for mask
			self.index_pid[index] = pid
			self.clothes_index[clothes_id].append(index)
			self.index_dic[pid].append(index)
			self.pid_clothes[pid].add(clothes_id)  # pid0:{0, 1, 2}
			self.pid_index[pid].append(index)

		# self.pid_clothes_index[pid].apend(index)
		# 对self.pid_clothes里面的元素进行替换,嵌套字典
		self.pid_clothes_index_only_one = {}  # 外层字典 # 先统计只有一件衣服的样本
		self.pid_clothes_index_more_than_one = {}  # 外层字典 # 统计有两件yishang
		for k, v in self.pid_clothes.items():  # 从这得到每个pid对应的衣服
			self.clothes_index_new = {}  # 内层字典
			# k=pids v=clothes
			if len(v) > 1:
				for v_i in v:
					index_i = self.clothes_index[v_i]
					self.clothes_index_new[v_i] = index_i
				self.pid_clothes_index_more_than_one[k] = self.clothes_index_new
			else:
				for v_i in v:
					index_i = self.clothes_index[v_i]
					self.clothes_index_new[v_i] = index_i
				self.pid_clothes_index_only_one[k] = self.clothes_index_new
		self.pids = list(self.index_dic.keys())
		self.num_identities = len(self.pids)  # 77

		# 处理只有一件衣服的情况,统计一下.由于只有一件衣服,那么只能按照正常的采样
		self.length = 0
		self.pids_only_one = list(self.pid_clothes_index_only_one.keys())
		# for pid_one in self.pids_only_one:
		# 	one_clothes_index_idsx = self.pid_clothes_index_only_one[pid_one].values()
		# 	num_one = len(list(one_clothes_index_idsx)[0])
		# 	if num_one < self.num_instances:
		# 		num_one = self.num_instances
		# self.length += num_one - num_one % self.num_instances

		# 处理有超过2件衣服情况
		self.pids_morethan_one = list(self.pid_clothes_index_more_than_one.keys())
		# for pid_more_one in self.pids_morethan_one:
		# 	more_than_one_clothes_dict = self.pid_clothes_index_more_than_one[pid_more_one]
		# 	for k3, v3 in more_than_one_clothes_dict.items():
		# 		num_two = len(v3)
		# 		if num_two < self.num_instances:
		# 			num_two = self.num_instances
		# 		if num_two < 2 * self.num_instances and num_two >= (1.5 * self.num_instances):
		# 			num_two = 2 * self.num_instances
		# self.length += num_two - num_two % self.num_instances

		# assert self.length % self.num_instances == 0  # 8976 // 2 = 4488

		# if self.length // self.num_instances % self.num_replicas != 0:
		# 	self.num_samples = math.ceil(
		# 		(self.length // self.num_instances - self.num_replicas) / self.num_replicas) * self.num_instances
		# else:
		# 	self.num_samples = math.ceil(self.length / self.num_replicas)
		self.total_size = 0

	def __iter__(self):
		# deterministically shuffle based on epoch and seed
		random.seed(self.seed + self.epoch)
		np.random.seed(self.seed + self.epoch)

		batch_idxs_dict = defaultdict(list)  # 对于只有一件衣服的样本而言,直接按照正常的采样就行.
		for pid_1 in self.pids_only_one:
			one_clothes_index_idsx = copy.deepcopy(list(self.pid_clothes_index_only_one[pid_1].values())[0])
			if len(one_clothes_index_idsx) < self.num_instances:
				one_clothes_index_idsx = np.random.choice(one_clothes_index_idsx, size=self.num_instances, replace=True)
			random.shuffle(one_clothes_index_idsx)
			batch_idx_1 = []
			for idx_1 in one_clothes_index_idsx:
				batch_idx_1.append(idx_1)
				if len(batch_idx_1) == self.num_instances:
					batch_idxs_dict[pid_1].append(batch_idx_1)
					batch_idx_1 = []

		pid_more_than_one_dict = copy.deepcopy(self.pid_clothes_index_more_than_one)
		for pid_2 in self.pids_morethan_one:
			more_than_1_clothes_dict = pid_more_than_one_dict[pid_2]
			pid_2_clothes_id_list = list(more_than_1_clothes_dict.keys())  # 衣服id列表 [0, 1]
			avai_clothes_id = pid_2_clothes_id_list  # avai_clothes_id = copy.deepcopy(pid_2_clothes_id_list)
			while len(avai_clothes_id) > 1:
				selected_clothes_ids = random.sample(avai_clothes_id, 1)  # 随机选1件衣服id
				index_c_1 = more_than_1_clothes_dict[selected_clothes_ids[0]]
				random.shuffle(index_c_1)
				if len(index_c_1) >= 8:
					selected_8_index_c1 = [index_c_1.pop(0) for i in range(8)]
				else:
					selected_8_index_c1 = []
					wuyong = [index_c_1.pop(0) for i in range(len(index_c_1))]
					avai_clothes_id.remove(selected_clothes_ids[0])

				if len(selected_8_index_c1) == 0:
					continue
				else:
					selected_8_index = selected_8_index_c1
					batch_idxs_dict[pid_2].append(selected_8_index)
			# batch_idx_2.append(selected_4_index)
			# 如果还剩下一个一件衣服,
			while len(avai_clothes_id) > 0:
				index_c_remaind = more_than_1_clothes_dict[avai_clothes_id[0]]
				if len(index_c_remaind) >= 8:
					selected_8_index_remaind = [index_c_remaind.pop(0) for i in range(8)]
				else:
					selected_8_index_remaind = []
					wuyong = [index_c_remaind.pop(0) for i in range(len(index_c_remaind))]
					avai_clothes_id.remove(avai_clothes_id[0])
				if len(selected_8_index_remaind) == 0:
					continue
				else:
					batch_idxs_dict[pid_2].append(selected_8_index_remaind)

		# pk sampling
		avai_pids_final = copy.deepcopy(self.pids)
		final_idxs = []

		while len(avai_pids_final) >= 4:
			batch_idx_22 = []
			selected_pids = random.sample(avai_pids_final, 4)
			for pid_f in selected_pids:
				batch_idxs = batch_idxs_dict[pid_f].pop(0)
				batch_idx_22.extend(batch_idxs)
				if len(batch_idxs_dict[pid_f]) == 0:
					avai_pids_final.remove(pid_f)
			final_idxs.append(batch_idx_22)

		# remove tail of data to make it evenly divisible.
		# list_container = final_idxs[: total_size//self.num_instances]
		# assert len(list_container) == self.total_size//self.num_instances
		random.shuffle(final_idxs)
		if len(final_idxs) % self.num_replicas > 0:
			final_idxs.pop(0)

		num_pk = len(final_idxs)

		assert len(final_idxs) % self.num_replicas == 0
		# subsample
		list_container = final_idxs[self.rank:num_pk:self.num_replicas]
		# assert len(list_container) == self.num_samples//self.num_instances

		ret = []
		for batch_idxs in list_container:
			ret.extend(batch_idxs)

		return iter(ret)

	@property
	def num_samples2(self):
		# deterministically shuffle based on epoch and seed
		random.seed(self.seed + self.epoch)
		np.random.seed(self.seed + self.epoch)

		batch_idxs_dict = defaultdict(list)  # 对于只有一件衣服的样本而言,直接按照正常的采样就行.
		for pid_1 in self.pids_only_one:
			one_clothes_index_idsx = copy.deepcopy(list(self.pid_clothes_index_only_one[pid_1].values())[0])
			if len(one_clothes_index_idsx) < self.num_instances:
				one_clothes_index_idsx = np.random.choice(one_clothes_index_idsx, size=self.num_instances, replace=True)
			random.shuffle(one_clothes_index_idsx)
			batch_idx_1 = []
			for idx_1 in one_clothes_index_idsx:
				batch_idx_1.append(idx_1)
				if len(batch_idx_1) == self.num_instances:
					batch_idxs_dict[pid_1].append(batch_idx_1)
					batch_idx_1 = []

		pid_more_than_one_dict = copy.deepcopy(self.pid_clothes_index_more_than_one)
		for pid_2 in self.pids_morethan_one:
			more_than_1_clothes_dict = pid_more_than_one_dict[pid_2]
			pid_2_clothes_id_list = list(more_than_1_clothes_dict.keys())  # 衣服id列表 [0, 1]
			avai_clothes_id = pid_2_clothes_id_list  # avai_clothes_id = copy.deepcopy(pid_2_clothes_id_list)
			while len(avai_clothes_id) > 1:
				selected_clothes_ids = random.sample(avai_clothes_id, 1)  # 随机选1件衣服id
				index_c_1 = more_than_1_clothes_dict[selected_clothes_ids[0]]
				random.shuffle(index_c_1)
				if len(index_c_1) >= 8:
					selected_8_index_c1 = [index_c_1.pop(0) for i in range(8)]
				else:
					selected_8_index_c1 = []
					wuyong = [index_c_1.pop(0) for i in range(len(index_c_1))]
					avai_clothes_id.remove(selected_clothes_ids[0])

				if len(selected_8_index_c1) == 0:
					continue
				else:
					selected_8_index = selected_8_index_c1
					batch_idxs_dict[pid_2].append(selected_8_index)
			# batch_idx_2.append(selected_4_index)
			# 如果还剩下一个一件衣服,
			while len(avai_clothes_id) > 0:
				index_c_remaind = more_than_1_clothes_dict[avai_clothes_id[0]]
				if len(index_c_remaind) >= 8:
					selected_8_index_remaind = [index_c_remaind.pop(0) for i in range(8)]
				else:
					selected_8_index_remaind = []
					wuyong = [index_c_remaind.pop(0) for i in range(len(index_c_remaind))]
					avai_clothes_id.remove(avai_clothes_id[0])
				if len(selected_8_index_remaind) == 0:
					continue
				else:
					batch_idxs_dict[pid_2].append(selected_8_index_remaind)

		# pk sampling
		avai_pids_final = copy.deepcopy(self.pids)
		final_idxs = []

		while len(avai_pids_final) >= 4:
			batch_idx_22 = []
			selected_pids = random.sample(avai_pids_final, 4)
			for pid_f in selected_pids:
				batch_idxs = batch_idxs_dict[pid_f].pop(0)
				batch_idx_22.extend(batch_idxs)
				if len(batch_idxs_dict[pid_f]) == 0:
					avai_pids_final.remove(pid_f)
			final_idxs.append(batch_idx_22)

		# remove tail of data to make it evenly divisible.
		# list_container = final_idxs[: total_size//self.num_instances]
		# assert len(list_container) == self.total_size//self.num_instances

		if len(final_idxs) % self.num_replicas > 0:
			final_idxs.pop(0)

		ret = []
		for batch_idxs in final_idxs:
			ret.extend(batch_idxs)

		total_size = len(ret) // 2

		return total_size

	def __len__(self):

		return self.num_samples2

	def set_epoch(self, epoch):
		"""Sets the epoch for this sampler. This ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
		self.epoch = epoch


class DistributedInferenceSampler(Sampler):
	"""
    refer to: https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py

    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

	def __init__(self, dataset, rank=None, num_replicas=None):
		if num_replicas is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			num_replicas = dist.get_world_size()  # 2
		if rank is None:
			if not dist.is_available():
				raise RuntimeError("Requires distributed package to be available")
			rank = dist.get_rank()  # 0
		self.dataset = dataset
		self.num_replicas = num_replicas  # 2
		self.rank = rank  # 0

		self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))  # 247
		self.total_size = self.num_samples * self.num_replicas  # 494

	def __iter__(self):
		indices = list(range(len(self.dataset)))
		# add extra samples to make it evenly divisible
		indices += [indices[-1]] * (self.total_size - len(indices))
		# subsample
		indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
		return iter(indices)

	def __len__(self):
		return self.num_samples