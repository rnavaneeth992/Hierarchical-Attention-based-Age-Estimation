import os
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Datasets.UTKFace.UTKFaceClassifierDataset import UTKFaceClassifierDataset
from Datasets.UTKFace.DataParser import DataParser
from Datasets.UTKFace.UTKFaceClassifierDataset import UTKFaceClassifierDataset
from Losses.MeanVarianceLoss import MeanVarianceLoss
from Models.JoinedTransformerModel import JoinedTransformerModel
from Models.UnifiedClassificationAndRegressionAgeModel import UnifiedClassificationAndRegressionAgeModel
from Models.transformer import *
from Models.unified_transformer_model import AgeTransformer
from Optimizers.RangerLars import RangerLars
from Schedulers.GradualWarmupScheduler import GradualWarmupScheduler
from Training.train_unified_model_iter import train_unified_model_iter


def get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size):
	pretrained_model = UnifiedClassificationAndRegressionAgeModel(7, 10, 15, 80)
	pretrained_model_path = 'weights'

	pretrained_model_file = os.path.join(pretrained_model_path, "unified_class_and_regress.pt")
	pretrained_model.load_state_dict(torch.load(pretrained_model_file), strict=False)

	num_features = pretrained_model.num_features
	backbone = pretrained_model.base_net
	backbone.train()
	backbone.to(device)

	transformer = TransformerModel(
		age_interval, min_age, max_age,
		mid_feature_size, mid_feature_size,
		num_outputs=num_classes,
		n_heads=4, n_encoders=4, dropout=0.3,
		mode='mean').to(device)
	age_transformer = AgeTransformer(backbone, transformer, num_features, mid_feature_size).to(device)


	return age_transformer

	model = JoinedTransformerModel(num_classes, age_interval, min_age, max_age, device, mid_feature_size)
	model.to(device)
	model.train()

	return model


if __name__ == "__main__":
	seed = 42
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	torch.cuda.empty_cache()

	torch.backends.cudnn.benchmark = True

	min_age = 1
	max_age = 100
	age_interval = 1 
	batch_size = 8
	num_iters = int(1.5e5)
	random_split = True
	num_copies = 10
	mid_feature_size = 1024

	num_classes = int((max_age - min_age) / age_interval + 1)

	data_parser = DataParser('./Datasets/aligned_dataset.hdf5')
	data_parser.initialize_data()

	x_train, y_train, x_test, y_test = data_parser.x_train,	data_parser.y_train, data_parser.x_test, data_parser.y_test,

	if random_split:
		all_images = np.concatenate((x_train, x_test), axis=0)
		all_labels = np.concatenate((y_train, y_test), axis=0)

		x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.20, random_state=42)

	train_ds = ARClassifierDataset(
		x_train,
		y_train,
		min_age,
		age_interval,
		transforms.Compose([
			transforms.RandomResizedCrop(224, (0.9, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomApply([transforms.ColorJitter(
				brightness=0.1,
				contrast=0.1,
				saturation=0.1,
				hue=0.1
			)], p=0.5),
			transforms.RandomApply([transforms.RandomAffine(
				degrees=10,
				translate=(0.1, 0.1),
				scale=(0.9, 1.1),
				shear=5,
				resample=Image.BICUBIC
			)], p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.RandomErasing(p=0.5)
		]),
		copies=num_copies
	)

	test_ds = ARClassifierDataset(
		x_test,
		y_test,
		min_age,
		age_interval,
		transform=transforms.Compose([
			transforms.RandomResizedCrop(224, (0.9, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomApply([transforms.ColorJitter(
				brightness=0.1,
				contrast=0.1,
				saturation=0.1,
				hue=0.1
			)], p=0.5),
			transforms.RandomApply([transforms.RandomAffine(
				degrees=10,
				translate=(0.1, 0.1),
				scale=(0.9, 1.1),
				shear=5,
				resample=Image.BICUBIC
			)], p=0.5),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.RandomErasing(p=0.5)
		]),
		copies=num_copies
	)

	image_datasets = {
		'train': train_ds,
		'val': test_ds
	}

	data_loaders = {
		'train': DataLoader(train_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True, drop_last=True),
		'val': DataLoader(test_ds, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False, drop_last=True)
	}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

	model = get_age_transformer(device, num_classes, age_interval, min_age, max_age, mid_feature_size)

	criterion_reg = nn.MSELoss().to(device)
	criterion_cls = torch.nn.CrossEntropyLoss().to(device)
	mean_var_criterion = MeanVarianceLoss(0, num_classes, device, lambda_mean=0.2, lambda_variance=0.05).to(device)

	optimizer = RangerLars(model.parameters(), lr=1e-3)

	num_epochs = int(num_iters / len(data_loaders['train'])) + 1

	cosine_scheduler = CosineAnnealingLR(
		optimizer,
		T_max=num_iters
	)
	scheduler = GradualWarmupScheduler(
		optimizer,
		multiplier=1,
		total_epoch=10000,
		after_scheduler=cosine_scheduler
	)

	model_path = 'weights/'
	if not os.path.exists(model_path):
		os.makedirs(model_path)

	best_model = train_unified_model_iter(
		model,
		criterion_reg,
		criterion_cls,
		mean_var_criterion,
		optimizer,
		scheduler,
		data_loaders,
		dataset_sizes,
		device,
		writer,
		model_path,
		num_classes,
		num_epochs=num_epochs,
		validate_at_k=1000)

	print('saving best model')

	FINAL_MODEL_FILE = os.path.join(model_path, "age_estimator.pt")
	torch.save(best_model.state_dict(), FINAL_MODEL_FILE)
