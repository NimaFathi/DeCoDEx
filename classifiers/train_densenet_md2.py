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

from core.image_datasets import MedicalDeviceDataset
from torch.nn import functional as F

#DenseNet121
class DenseNet121(nn.Module):
	def __init__(self, num_classes):
		"""
		Init model architecture
		
		Parameters
		----------
		num_classes: int
			number of classes
		is_trained: bool
			whether using pretrained model from ImageNet or not
		"""
		super().__init__()
		
		# Load the DenseNet121 from ImageNet
		self.net = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
		
		# Get the input dimension of last layer
		kernel_count = self.net.classifier.in_features
		
		# Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
		self.net.classifier = nn.Linear(kernel_count, num_classes)
		
	def forward(self, inputs):
		"""
		Forward the netword with the inputs
		"""
		return self.net(inputs)


# #metric computation
# def multi_label_auroc(y_gt, y_pred):
#     """ Calculate AUROC for each class

#training epoch
def epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb):
	"""
	Epoch training

	Paramteters
	-----------
	epoch: int
	  epoch number
	model: torch Module
	  model to train
	train_dataloader: Dataset
	  data loader for training
	device: str
	  "cpu" or "cuda"
	loss_criteria: loss function
	  loss function used for training
	optimizer: torch optimizer
	  optimizer used for training
	mb: master bar of fastprogress
	  progress to log

	Returns
	-------
	float
	  training loss
	"""
	# Switch model to training mode
	model.train()
	training_loss = 0 # Storing sum of training losses
   
	# For each batch
	for batch, (images, labels) in enumerate(progress_bar(train_dataloader, parent=mb)):
		
		# Move X, Y  to device (GPU)
		images = images.to(device)
		labels = labels.to(device).float()
		
		# Clear previous gradient
		optimizer.zero_grad()

		# Feed forward the model
		pred = model(images)
		pred = pred.squeeze(1)
		loss = loss_criteria(pred, labels)

		# Back propagation
		loss.backward()

		# Update parameters
		optimizer.step()

		# Update training loss after each batch
		training_loss += loss.item()

		mb.child.comment = f'Training loss {training_loss/(batch+1)}'

	del images, labels, loss
	if torch.cuda.is_available(): torch.cuda.empty_cache()

	# return training loss
	return training_loss/len(train_dataloader)

def evaluating(epoch, model, val_loader, device, loss_criteria, mb):
	"""
	Validate model on validation dataset
	
	Parameters
	----------
	epoch: int
		epoch number
	model: torch Module
		model used for validation
	val_loader: Dataset
		data loader of validation set
	device: str
		"cuda" or "cpu"
	loss_criteria: loss function
	  loss function used for training
	mb: master bar of fastprogress
	  progress to log
  
	Returns
	-------
	float
		loss on validation set
	float
		metric score on validation set
	"""

	# Switch model to evaluation mode
	model.eval()

	val_loss = 0                                   # Total loss of model on validation set
	out_pred = torch.FloatTensor().to(device)      # Tensor stores prediction values
	out_gt = torch.FloatTensor().to(device)        # Tensor stores groundtruth values

	with torch.no_grad(): # Turn off gradient
		# For each batch
		for step, (images, labels) in enumerate(progress_bar(val_loader, parent=mb)):
			# Move images, labels to device (GPU)
			images = images.to(device)
			labels = labels.to(device).float()

			# Update groundtruth values
			out_gt = torch.cat((out_gt,  labels), 0)

			# Feed forward the model
			ps = model(images)
			ps = ps.squeeze(1)
			loss = loss_criteria(ps, labels)

			# Update prediction values
			out_pred = torch.cat((out_pred, ps), 0)

			# Update validation loss after each batch
			val_loss += loss
			mb.child.comment = f'Validation loss {val_loss/(step+1)}'

	# Clear memory
	del images, labels, loss
	if torch.cuda.is_available(): torch.cuda.empty_cache()
	# return validation loss, and metric score
	return val_loss/len(val_loader), np.array(roc_auc_score(out_gt.to("cpu").numpy(), out_pred.to("cpu").numpy())).mean()

if __name__ == "__main__":
	# ============================================================================
	# Argument Parsing
	# ============================================================================

	parser = argparse.ArgumentParser(description="Training a classifier/detector for contrived CheXpert/Pleural Effusion dataset")
	parser.add_argument("--data_dir", type=str, default='/usr/local/faststorage/datasets/chexpert', help="Directory of the dataset")
	parser.add_argument("--model_path", type=str, default="pretrained/md_detection_densenet121_256_lotits_all.pth", help="Path to save the model")
	parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
	parser.add_argument("--image_size", type=int, default=256, help="Size of the images")
	parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
	parser.add_argument('--lr_sf', type=float, default=0.1, help='Learning rate scheduler factor')
	parser.add_argument('--lr_patience', type=int, default=5, help='Learning rate scheduler patience')
	parser.add_argument("--task", type=str, default="detection", choices=["classification", "detection"],
						help="Task for the model")
	parser.add_argument("--random_crop", action="store_true", help="Use random cropping as augmentation")
	parser.add_argument("--random_flip", action="store_true", help="Use random flipping as augmentation")
	parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
	args = parser.parse_args()
	args.model_path = f"pretrained/md_{args.task}_densenet121_{args.image_size}_logits_all2.pth"
	
	# Device
	if torch.cuda.is_available():
		device = torch.device(f"cuda:{args.gpu_id}")
	else:
		device = torch.device("cpu")


	#define a model 
	# usind num_classes = 1 for binary classification
	model = DenseNet121(num_classes=1)
	model = model.to(device)

	#define loss function
	# Loss function
	loss_criteria = nn.BCEWithLogitsLoss()

	# Adam optimizer
	optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

	# Learning rate will be reduced automatically during training
	lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = args.lr_sf, patience = args.lr_patience, mode = 'min', verbose=True)
	dataset_train = MedicalDeviceDataset(image_size=args.image_size,
										  data_dir=args.data_dir,
										  partition='train',
										  task=args.task, # 'classification' or 'detection
										  random_crop=args.random_crop,
										  random_flip=args.random_flip,
										  csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
										  )
	
	dataset_val = MedicalDeviceDataset(image_size=args.image_size,
										data_dir=args.data_dir,
										partition='val',
										task=args.task, # 'classification' or 'detection
										random_crop=False,
										random_flip=False,
										csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
										)
	
	print(f'Images on the dataset:{len(dataset_train)} [Train], {len(dataset_val)} [Val]', )

	train_dataloader = data.DataLoader(dataset_train, batch_size=args.batch_size,
								shuffle=True,
								num_workers=4, pin_memory=True)
	val_dataloader = data.DataLoader(dataset_val, batch_size=args.batch_size,
								shuffle=False,
								num_workers=4, pin_memory=True)
	# Best AUROC value during training
	best_score = 0
	training_losses = []
	validation_losses = []
	validation_score = []


	# Config progress bar
	mb = master_bar(range(args.epochs))
	mb.names = ['Training loss', 'Validation loss', 'Validation AUROC']
	x = []

	nonimproved_epoch = 0

	# Training each epoch
	for epoch in mb:
		mb.comment = f'Best AUROC score: {best_score}'
		x.append(epoch)

		# Training
		train_loss = epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb)
		mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))
		training_losses.append(train_loss)

		# Evaluating
		val_loss, new_score = evaluating(epoch, model, val_dataloader, device, loss_criteria, mb)
		mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, new_score))
		validation_losses.append(val_loss)
		validation_score.append(new_score)

		# Update learning rate
		lr_scheduler.step(val_loss)

		# Update training chart
		mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0,epoch+1], [0,1])

		# Save model
		if best_score < new_score:
			mb.write(f"Improve AUROC from {best_score} to {new_score}")
			best_score = new_score
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