from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torch
num_classes = 10 
batch_size = 32 
# 方案 1：基于类别权重的采样器
class_sample_counts = [len(train_dataset.targets[train_dataset.targets == c]) for c in range(num_classes)]
class_weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
sample_weights = class_weights[train_dataset.targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
# 方案 2：基于类别权重的损失函数
loss = nn.CrossEntropyLoss(weight=class_weights.to(device))
