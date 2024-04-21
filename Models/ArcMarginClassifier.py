import math

import torch
from torch import nn as nn
from torch.nn import Linear, functional as F
from torchvision.models import resnet34, vgg16


class ArcMarginClassifier(nn.Module):
	def __init__(self, num_classes, margin_m=0.5, margin_s=64, easy_margin=False):
		super(ArcMarginClassifier, self).__init__()

		self.Labels = range(num_classes)
		k = 512
		self.easy_margin = easy_margin
		self.m = margin_m
		self.s = margin_s

		self.cos_m = math.cos(self.m)
		self.sin_m = math.sin(self.m)
		self.th = math.cos(math.pi - self.m)
		self.mm = math.sin(math.pi - self.m) * self.m

		self.base_net = vgg16(pretrained=True)
		self.base_net.classifier[6] = nn.Linear(4096, k)

		self.weight = nn.Parameter(torch.FloatTensor(num_classes, k))
		nn.init.xavier_uniform_(self.weight)

	def freeze_base_cnn(self, should_freeze=True):
		for param in self.base_net.parameters():
			param.requires_grad = not should_freeze

	def forward(self, input_images, labels=None, device=torch.device("cuda:0")):
		x = self.base_net(input_images)

		x = F.normalize(x)
		W = F.normalize(self.weight)

		cosine = F.linear(x, W)

		if labels is None:
			return cosine * self.s

		sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
		phi = cosine * self.cos_m - sine * self.sin_m 

		if self.easy_margin:
			phi = torch.where(cosine > 0, phi, cosine)
		else:
			phi = torch.where(cosine > self.th, phi, cosine - self.mm)

		one_hot = torch.zeros(cosine.size(), device=device)
		one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		output *= self.s

		return output