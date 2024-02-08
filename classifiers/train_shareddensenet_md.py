##############################################################################################################
##############  Train the classifier for chexpert using Densenet-121                      ####################
##############################################################################################################
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from fastprogress import master_bar, progress_bar

import torch
import torch.nn as nn
from torch.utils import data
from torch import optim

import torchvision
from torchvision.models import DenseNet121_Weights
import torchvision.models as models

from core.image_datasets import MedicalDevicePEDataset, MedicalDeviceDataset
from torch.nn import functional as F

class TwoHeadedDenseNet(nn.Module):
	def __init__(self, num_classes=1):
		super(TwoHeadedDenseNet, self).__init__()
		self.backbone = models.densenet121(pretrained=True)
		num_features = self.backbone.classifier.in_features
		self.head1 = nn.Linear(num_features, num_classes)  # Classifier head
		self.head2 = nn.Linear(num_features, num_classes)  # Detector head

	def forward(self, x, head_idx):
		features = self.backbone.features(x)
		out = F.relu(features, inplace=True)
		out = F.adaptive_avg_pool2d(out, (1, 1))
		out = torch.flatten(out, 1)
		if head_idx == 0:
			out = self.head1(out)
		elif head_idx == 1:
			out = self.head2(out)
		return out


#training epoch
def epoch_training(epoch, model, train_dataloader_cl, train_dataloader_det, device, loss_criteria, optimizer, mb, mode='simultaneous'):
	"""
	Epoch training

	Paramteters
	-----------
	epoch: int
	  epoch number
	model: torch Module
	  model used for training
	train_dataloader: Dataset
	  data loader of training set one for classification and one for detection
	device: str
	  "cuda" or "cpu"
	loss_criteria: loss function
	  loss function used for training
	optimizer: optimizer
	  optimizer used for training
	focused_epochs: int
	  number of epochs for focused training
	mb: master bar of fastprogress
	  progress to log
	mode: str
	  'simultaneous' or 'focused'

	Returns
	-------
	float
	  training loss
	"""
	# Switch model to training mode
	print(f'training epoch {epoch} started')
	model.train()
	training_loss = 0 # Storing sum of training losses
	if mode == 'simultaneous':
		print('simultaneous')
		for batch, ((images_cls, labels_cls), (images_det, labels_det)) in enumerate(zip(train_dataloader_cl, train_dataloader_det)):
			images_cls, labels_cls = images_cls.to(device), labels_cls.to(device)
			images_det, labels_det = images_det.to(device), labels_det.to(device)
			optimizer.zero_grad()

			# Classification
			outputs_cls = model(images_cls, 0)
			loss_cls = loss_criteria(outputs_cls, labels_cls.float().unsqueeze(1))

			# Detection
			outputs_det = model(images_det, 1)
			loss_det = loss_criteria(outputs_det, labels_det.float().unsqueeze(1))

			# Combined loss
			loss = loss_cls + loss_det
			loss.backward()
			optimizer.step()
			training_loss += loss.item()

		del images_cls, images_det, labels_cls, labels_det, loss
		if torch.cuda.is_available(): torch.cuda.empty_cache()
		# Focused training phase
	else:	
		print('focused')
		for bx, images_cls, labels_cls in enumerate(progress_bar(train_dataloader_cl, parent=mb)):
			images_cls, labels_cls = images_cls.to(device), labels_cls.to(device)
			optimizer.zero_grad()
			outputs_cls = model(images_cls, 0)
			loss_cls = loss_criteria(outputs_cls, labels_cls.float().unsqueeze(1))
			loss_cls.backward()
			optimizer.step()
			training_loss += loss_cls.item()
			mb.child.comment = f'Training loss Classifier {training_loss/(bx+1)}'

		# Focus on detection
		for bx, images_det, labels_det in enumerate(progress_bar(train_dataloader_det, parent=mb)):
			images_det, labels_det = images_det.to(device), labels_det.to(device)
			optimizer.zero_grad()
			outputs_det = model(images_det, 1)
			loss_det = loss_criteria(outputs_det, labels_det.float().unsqueeze(1))
			loss_det.backward()
			optimizer.step()
			training_loss += loss_det.item()
			mb.child.comment = f'Training loss detector {training_loss/(bx+1)}'

		del images_cls, images_det, labels_cls, labels_det, loss
	return training_loss

	
def evaluating(epoch, model, val_dataloader_cl, val_dataloader_det, device, loss_criteria, mb, mode='simultaneous'):
	"""
	Epoch training

	Paramteters
	-----------
	epoch: int
	  epoch number
	model: torch Module
	  model used for training
	val_dataloader: Dataset
	  data loader of training set one for classification and one for detection
	device: str
	  "cuda" or "cpu"
	loss_criteria: loss function
	  loss function used for training
	focused_epochs: int
	  number of epochs for focused training
	mb: master bar of fastprogress
	  progress to log
	mode: str
	  'simultaneous' or 'focused'

	Returns
	-------
	float
	  	validation loss
		metric score on validation set
	"""

	# Switch model to evaluation mode
	model.eval()

	val_loss = 0                                   # Total loss of model on validation set
	out_pred_cls = torch.FloatTensor().to(device)      # Tensor stores prediction values
	out_gt_cls = torch.FloatTensor().to(device)        # Tensor stores groundtruth values
	out_pred_det = torch.FloatTensor().to(device)      # Tensor stores prediction values
	out_gt_det = torch.FloatTensor().to(device)        # Tensor stores groundtruth values

	with torch.no_grad(): # Turn off gradient
		for batch, (images_cls, labels_cls) in enumerate(progress_bar(val_dataloader_cl, parent=mb)):
			images_cls, labels_cls = images_cls.to(device), labels_cls.to(device)

			# Classification
			outputs_cls = model(images_cls, 0)
			# Update groundtruth values
			out_gt_cls = torch.cat((out_gt_cls,  labels_cls), 0)

			loss_cls = loss_criteria(outputs_cls, labels_cls.float().unsqueeze(1))

			# Update prediction values
			out_pred_cls = torch.cat((out_pred_cls, outputs_cls), 0)
			mb.child.comment = f'Validation loss Classifier {val_loss/(batch+1)}'

	with torch.no_grad():
		for bx, (images_det, labels_det) in enumerate(progress_bar(val_dataloader_det, parent=mb)):
			# Detection
			images_det, labels_det = images_det.to(device), labels_det.to(device)

			outputs_det = model(images_det, 1)
			loss_det = loss_criteria(outputs_det, labels_det.float().unsqueeze(1))

			out_gt_det = torch.cat((out_gt_det,  labels_det), 0)
			out_pred_det = torch.cat((out_pred_det, outputs_det), 0)


			# Combined loss
			loss = loss_cls + loss_det

			val_loss += loss.item()
			mb.child.comment = f'Validation loss Detector {val_loss/(batch+1)}'
			
		del images_cls, images_det, labels_cls, labels_det, loss
		if torch.cuda.is_available(): torch.cuda.empty_cache()
		# Focused training phase
	

	# return validation loss, and metric score
	return val_loss, np.array(roc_auc_score(out_gt_cls.to("cpu").numpy(), out_pred_cls.to("cpu").numpy())).mean(), np.array(roc_auc_score(out_gt_det.to("cpu").numpy(), out_pred_det.to("cpu").numpy())).mean()

if __name__ == "__main__":
	# ============================================================================
	# Argument Parsing
	# ============================================================================

	parser = argparse.ArgumentParser(description="Training a classifier/detector for contrived CheXpert/Pleural Effusion dataset")
	parser.add_argument("--data_dir", type=str, default='/usr/local/faststorage/datasets/chexpert', help="Directory of the dataset")
	parser.add_argument("--model_path", type=str, default="pretrained/md_shared_densenet121_256.pth", help="Path to save the model")
	parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
	parser.add_argument("--focused_epochs", type=int, default=10, help="Number of epochs for focused training")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
	parser.add_argument("--image_size", type=int, default=256, help="Size of the images")
	parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
	parser.add_argument('--lr_sf', type=float, default=0.1, help='Learning rate scheduler factor')
	parser.add_argument('--lr_patience', type=int, default=5, help='Learning rate scheduler patience')
	parser.add_argument("--task", type=str, default="classification", choices=["classification", "detection"],
						help="Task for the model")
	parser.add_argument("--random_crop", action="store_true", help="Use random cropping as augmentation")
	parser.add_argument("--random_flip", action="store_true", help="Use random flipping as augmentation")
	parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
	parser.add_argument("--biased", action="store_true", help="Use biased dataset")
	parser.add_argument("--balanced", action="store_true", help="Use balanced dataset")
	parser.add_argument("--balance_ratio", type=float, default=1, help="Ratio of positive samples in the balanced dataset")
	args = parser.parse_args()

	args.model_path = "pretrained/md_shared_densenet121_256.pth"
	# Device
	if torch.cuda.is_available():
		device = torch.device(f"cuda:{args.gpu_id}")
	else:
		device = torch.device("cpu")


	#define a model 
	# usind num_classes = 1 for binary classification
	model = TwoHeadedDenseNet().to(device)
	model = model.to(device)

	#define loss function
	# Loss function
	loss_criteria = nn.BCEWithLogitsLoss()

	# Adam optimizer
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

	# Learning rate will be reduced automatically during training
	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = args.lr_sf, patience = args.lr_patience, mode = 'min', verbose=True)
	dataset_train_cl = MedicalDevicePEDataset(image_size=args.image_size,
										  data_dir=args.data_dir,
										  partition='train',
										  task='classification', # 'classification' or 'detection
										  random_crop=True,
										  random_flip=True,
										  biased=args.biased,
										  rebalance=args.balanced,
										  csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
										  ratio=args.balance_ratio,
										  )
	
	dataset_val_cl = MedicalDevicePEDataset(image_size=args.image_size,
										data_dir=args.data_dir,
										partition='val',
										task='detection', # 'classification' or 'detection
										random_crop=False,
										random_flip=False,
										biased=args.biased,
										rebalance=args.balanced,
										csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
										ratio=args.balance_ratio
										)
	dataset_train_det = MedicalDeviceDataset(image_size=args.image_size,
										  data_dir=args.data_dir,
										  partition='train',
										  task='detection', # 'classification' or 'detection
										  random_crop=args.random_crop,
										  random_flip=args.random_flip,
										  csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
										  )
	
	dataset_val_det = MedicalDeviceDataset(image_size=args.image_size,
										data_dir=args.data_dir,
										partition='val',
										task="detection", # 'classification' or 'detection
										random_crop=False,
										random_flip=False,
										csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
										)
	print(f'Images on the dataset [Classification]:{len(dataset_train_cl)} [Train], {len(dataset_val_cl)} [Val]', )
	print(f'Images on the dataset [Classification]:{len(dataset_train_det)} [Train], {len(dataset_val_det)} [Val]', )


	train_dataloader_cl = data.DataLoader(dataset_train_cl, batch_size=args.batch_size,
								shuffle=True,
								num_workers=4, pin_memory=True)
	val_dataloader_cl = data.DataLoader(dataset_val_cl, batch_size=args.batch_size,
								shuffle=False,
								num_workers=4, pin_memory=True)
	
	train_dataloader_det = data.DataLoader(dataset_train_det, batch_size=args.batch_size,
								shuffle=True,
								num_workers=4, pin_memory=True)
	val_dataloader_det = data.DataLoader(dataset_val_det, batch_size=args.batch_size,
								shuffle=False,
								num_workers=4, pin_memory=True)
	
	# Best AUROC value during training
	best_score = 0
	training_losses = []
	validation_losses = []
	validation_score_cl = []
	validation_score_det = []


	# Config progress bar
	mb = master_bar(range(args.epochs))
	mb.names = ['Training loss simultaneous','Training loss Classifier', 'Training loss Detector', 'Validation loss', 'Validation loss Classifier', 'Validation loss Detector', 'Validation AUROC Classifier', 'Validation AUROC Detector']
	x = []

	nonimproved_epoch = 0
	# Training each epoch
	for epoch in mb:
		print('we are here')
		mb.comment = f'Best AUROC score: {best_score}'
		x.append(epoch)
		# Training
		if epoch < args.epochs - args.focused_epochs:
			print('expected')
			train_loss = epoch_training(epoch, model, train_dataloader_cl, train_dataloader_det, device, loss_criteria, optimizer, mb, mode='simultaneous')
			mb.write('Finish training epoch {}/{} with loss {:.4f}'.format(epoch, 'simultaneous', train_loss))
		else: 
			print('unexpected')
			train_loss = epoch_training(epoch, model, train_dataloader_cl, train_dataloader_det, device, loss_criteria, optimizer, mb, mode='focused')
			mb.write('Finish training epoch {}/{} with loss {:.4f}'.format(epoch, 'focused', train_loss))

		training_losses.append(train_loss)

		# Evaluating
		val_loss, new_score_cl, new_score_det = evaluating(epoch, model, val_dataloader_cl, val_dataloader_det, device, loss_criteria, mb)
		
		mb.write('Finish validation epoch {} with loss {:.4f} and score cls {:.4f} score det {:.4f}'.format(epoch, val_loss, new_score_cl, new_score_det))
		validation_losses.append(val_loss)
		validation_score_cl.append(new_score_cl)
		validation_score_det.append(new_score_det)

		# Update learning rate
		lr_scheduler.step(val_loss)

		# Update training chart
		mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score_cl], [x, val_dataloader_det]], [0,epoch+1], [0,1])
		avg_score = (10 * new_score_cl + new_score_det)/11
		# Save model
		if best_score < avg_score:
			mb.write(f"Improve AUROC from {best_score} to {avg_score}")
			best_score = avg_score
			nonimproved_epoch = 0
			torch.save({"model": model.state_dict(), 
						"optimizer": optimizer.state_dict(), 
						"best_score": best_score, 
						"epoch": epoch, 
						"lr_scheduler": lr_scheduler.state_dict()}, args.model_path)
		else: 
			nonimproved_epoch += 1
		if nonimproved_epoch > 10:
			mb.write(f"Early stopping at epoch {epoch}")
			break    
