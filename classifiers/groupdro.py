##############################################################################################################
##############  Train the classifier for chexpert using Densenet-121                      ####################
##############################################################################################################
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from fastprogress import master_bar, progress_bar
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
import os
import torch
from torch.utils.data import Dataset

from PIL import Image

import torchvision
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import roc_curve
from os import path as osp

import sys
sys.path.append('/usr/local/data/nimafh/midl2024-cfdiffusion')

from torch.nn import functional as F
from torchvision import transforms, datasets

def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate """
    fpr, tpr, threshold = roc_curve(target, predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold

class PE90DotNoSupportDatasetGroupDRO(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        partition,
        path='img_dot_healthy_90',
        task='classification', # it can either be classification or detection
        shard=0,
        num_shards=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        normalize=True,
        biased=False
    ):
        self.data_dir = data_dir
        data = pd.read_csv(osp.join(data_dir, 'info.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[data['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.class_cond = class_cond
        self.task = task
        self.path = path
        if biased:
            self.data = self.data[self.data['group'].isin([0, 3])]       # self.data.replace(-1, 0, inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        img_file = sample['Path']

        # Determine the label based on the task
        if self.task == 'classification':
            label = 0 if sample['group'] in [0, 2] else 1
        elif self.task == 'detection':
            # Convert 'Healthy/Unhealthy' to binary labels
            label = 1 if sample['group'] in [0,1] else 0
        elif self.task == 'both':
            det_label = 1 if sample['group'] in [0, 1] else 0
            cls_label = 0 if sample['group'] in [0, 2] else 1
        else:
            raise ValueError(f'Unknown task {self.task}')
        # Rest of the code to load and transform the image...
        with open(os.path.join(self.data_dir, self.path, img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        if self.task == 'both':
            return img, {'y': cls_label, 'z': det_label}
        return img, label, sample['group']




class MedicalDevicePEDatasetGroupDRO(Dataset):
    def __init__(
        self,
        image_size,
        data_dir,
        csv_dir,
        partition,
        path='img_chexpert',
        task='classification', # it can either be classification or detection
        shard=0,
        num_shards=1,
        ratio=1,
        class_cond=False,
        random_crop=True,
        random_flip=True,
        normalize=True,
        biased=False,
        rebalance=True,
        sample=False
    ):
        self.data_dir = data_dir
        if csv_dir is None:
            raise ValueError("unspecified csv directory")
        data = pd.read_csv(osp.join(csv_dir, 'list_attr_md.csv'))

        if partition == 'train':
            partition = 0
        elif partition == 'val':
            partition = 1
        elif partition == 'test':
            partition = 2
        else:
            raise ValueError(f'Unkown partition {partition}')

        self.data = data[data['partition'] == partition]
        self.data = self.data[shard::num_shards]
        self.data.reset_index(inplace=True)
        self.data.replace(-1, 0, inplace=True)

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

        self.class_cond = class_cond
        self.task = task
        self.path = path
        self.biased = biased
        if rebalance:
            self.data = self._balance_subjects(self.data, ratio=ratio, sample=sample)
    @staticmethod
    def _balance_subjects(df, ratio=1., sample=False):
        if ratio==1:
            return df
        else:
            if sample:
                gr_0 = df[df['group']==0].sample(frac=0.1)
                gr_3 = df[df['group']==3].sample(frac=0.1)
            else:
                gr_0 = df[df['group']==0]
                gr_3 = df[df['group']==3]
            majority_size = min(len(gr_0), len(gr_3))
            gr_0 = gr_0[:majority_size]
            gr_3 = gr_3[:majority_size]
            r1 = int(ratio * majority_size)
            r2 = int(ratio * majority_size)
            gr_1 = df[df['group']==1][:r2]
            gr_2 = df[df['group']==2][:r1]
            print(len(gr_0), len(gr_1), len(gr_2), len(gr_3))
            return pd.concat([gr_0, gr_1, gr_2, gr_3],axis=0)   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx, :]
        img_file = sample['Path']

        # Determine the label based on the task
        if self.task == 'classification':
                # Convert 'Healthy/Unhealthy' to binary labels make sure to change back to [0, 2]
                label = 0 if sample['group'] in [0, 2] else 1 
        elif self.task == 'detection':
            # Binary label for dot detection based on 'group'
            label = 1 if sample['group'] in [0, 1] else 0
        elif self.task == 'both':
            det_label = 1 if sample['group'] in [0, 1] else 0
            cls_label = 0 if sample['group'] in [0, 2] else 1 
        else:
            raise ValueError(f'Unknown task {self.task}')

        # Rest of the code to load and transform the image...
        with open(os.path.join(self.data_dir, self.path, img_file), "rb") as f:
            img = Image.open(f)
            img = img.convert('RGB')

        img = self.transform(img)
        if self.task == 'both':
            return img, {'y': cls_label, 'z': det_label}
        return img, label, sample['group']

class AugmentedMD(Dataset):
    def __init__(self, root_dir, random_flip=True, random_crop=True, normalize=True, image_size=256):
        """
        Args:
            root_dir (string): Directory with all the images and info files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.samples = self._load_samples()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip() if random_flip else lambda x: x,
            transforms.CenterCrop(image_size),
            transforms.RandomResizedCrop(image_size, (0.95, 1.0)) if random_crop else lambda x: x,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]) if normalize else lambda x: x
        ])

    def _load_samples(self):
        samples = []
        for cc_icc_folder in ['CC', ]:
            for ccf_icf_folder in ['CCF', 'ICF']:
                for cd_id_folder in ['CD', 'ID']:
                    ce_dir = os.path.join(self.root_dir, cc_icc_folder, ccf_icf_folder, cd_id_folder, 'CF')
                    info_dir = os.path.join(self.root_dir, cc_icc_folder, ccf_icf_folder, cd_id_folder, 'Info')
                    if os.path.isdir(ce_dir) and os.path.isdir(info_dir):
                        image_files = sorted([f for f in os.listdir(ce_dir) if f.endswith('.jpg')])
                        for image_file in image_files:
                            image_path = os.path.join(ce_dir, image_file)
                            info_path = os.path.join(info_dir, image_file.replace('.jpg', '.txt'))
                            if os.path.exists(info_path):
                                samples.append((image_path, info_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, info_path = self.samples[idx]
        image = Image.open(image_path).convert('RGB')  # Load the image
        
        # Initialize label value
        label = None
        
        # Read and parse the info file
        with open(info_path, 'r') as f:
            for line in f:
                if line.startswith('cf pred:'):
                    _, label_str = line.split(':')
                    label = int(label_str.strip())  # Convert the class pred value to an integer
                    break  # Exit the loop once the label is found
        
        if label is None:
            raise ValueError(f'Label not found in file: {info_path}')
        
        if self.transform:
            image = self.transform(image)  # Apply any transformations
        group = 0 if label == 2 else 1
        return image, label, group

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
    for batch, (images, labels, groups) in enumerate(progress_bar(train_dataloader, parent=mb)):
        
        # Move X, Y  to device (GPU)
        images = images.to(device)
        labels = labels.to(device).float()
        groups = groups.to(device).long()
        
        # Clear previous gradient
        optimizer.zero_grad()

        # Feed forward the model
        pred = model(images)
        pred = pred.squeeze(1)
        loss = loss_criteria(F.sigmoid(pred), labels, groups)

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
        for step, (images, labels, groups) in enumerate(progress_bar(val_loader, parent=mb)):
            # Move images, labels to device (GPU)
            images = images.to(device)
            labels = labels.to(device).float()
            groups = groups.to(device).long()

            # Update groundtruth values
            out_gt = torch.cat((out_gt,  labels), 0)

            # Feed forward the model
            ps = model(images)
            ps = ps.squeeze(1)
            loss = loss_criteria(F.sigmoid(ps), labels, groups)

            # Update prediction values
            out_pred = torch.cat((out_pred, ps), 0)

            # Update validation loss after each batch
            val_loss += loss
            mb.child.comment = f'Validation loss {val_loss/(step+1)}'
    cls_threshold = find_optimal_cutoff(out_gt.to("cpu").numpy(), out_pred.to("cpu").numpy())
    cls_binary_predictions = (out_pred > cls_threshold).float()
    # Clear memory
    del images, labels, loss
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    cls_roc_auc = roc_auc_score(out_gt.to("cpu").numpy(), cls_binary_predictions.to("cpu").numpy())
    # return validation loss, and metric score
    return val_loss/len(val_loader), cls_threshold, cls_roc_auc


class GroupDRO():
    def __init__(self, num_groups, eta, crit, device):
        """
        Calculates GroupDRO loss: https://arxiv.org/pdf/1911.08731.pdf
        Code Modified from: https://github.com/facebookresearch/BalancingGroups/blob/main/models.py#L218
        Main modification: -- in the original implementation gruops (g) are indexed separately from labels (y), using g and y all_g is calculated
                           -- in this implementation we are considering groups (g) to be all_g already
        Args:
            num_groups (int): total number of groups in the dataset
            eta (float): value of eta for updating q in GroupDRO (Algorithm-1: https://arxiv.org/pdf/1911.08731.pdf)
            crit (torch crit): orignal loss function - usually binary or categorical cross entropy
            device (torch.device): cuda or cpu device
        """
        super(GroupDRO, self).__init__()
        self.q = torch.ones(num_groups).to(device)
        self.q /= self.q.sum()
        self.eta = eta
        self.crit = crit
    def groups_(self, g):
        idx_group, idx_batch = [], []
        for grp in g.unique():
            idx_group.append(grp) # add grp
            idx_batch.append(g == grp) # in the batch make elements with g == grp to True and Flase otherwise
        return zip(idx_group, idx_batch)
    def __call__(self, y_hat, y, g):

        losses = self.crit(y_hat, y)
        for idx_g, idx_b in self.groups_(g):
            self.q[idx_g] *= (self.eta * losses[idx_b].mean()).exp().item()
        # normalize q
        self.q /= self.q.sum()
        # calculate actual losses
        loss_value = 0
        for idx_g, idx_b in self.groups_(g):
            loss_value += self.q[idx_g] * losses[idx_b].mean()
        return loss_value


if __name__ == "__main__":
    # ============================================================================
    # Argument Parsing
    # ============================================================================

    parser = argparse.ArgumentParser(description="Training a classifier/detector for contrived CheXpert/Pleural Effusion dataset")
    parser.add_argument("--data_dir", type=str, help="Directory of the dataset", required=True)
    parser.add_argument("--model_path", type=str, help="Path to save the model", required=True)
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the images")
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--lr_sf', type=float, default=0.1, help='Learning rate scheduler factor')
    parser.add_argument('--lr_patience', type=int, default=5, help='Learning rate scheduler patience')
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "detection"],
                        help="Task for the model")
    parser.add_argument("--random_crop", action="store_true", help="Use random cropping as augmentation")
    parser.add_argument("--random_flip", action="store_true", help="Use random flipping as augmentation")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--biased", action="store_true", help="Use biased dataset")
    parser.add_argument("--balanced", action="store_false", help="Use balanced dataset")
    parser.add_argument("--balance_ratio", type=float, default=0.1, help="Ratio of positive samples in the balanced dataset")
    parser.add_argument("--augment", action="store_true", help="Use augmented dataset")
    parser.add_argument("--augmented_data_dir", type=str, default="/usr/local/data/nimafh/midl2024-cfdiffusion/final_results/SD/MD_NoFinding256_gradreversal_final", help="Path to augmented data")
    parser.add_argument("--dataset", type=str, default="PE90DotNoSupport", help="Dataset to use", choices=["PE90DotNoSupport", "MedicalDevicePEDataset"])
    args = parser.parse_args()

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
    loss = nn.BCELoss(reduction='none')
    loss_criteria = GroupDRO(num_groups=4, eta=0.3, crit=loss, device=device)

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    # Learning rate will be reduced automatically during training
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = args.lr_sf, patience = args.lr_patience, mode = 'max', verbose=True)
    
    if args.dataset == 'PE90DotNoSupport':
        dataset_train_real = PE90DotNoSupportDatasetGroupDRO(image_size=args.image_size,
                                          data_dir=args.data_dir,
                                          partition='train',
                                          task=args.task, # 'classification' or 'detection
                                          random_crop=args.random_crop,
                                          random_flip=args.random_flip,
                                          biased=args.biased,
                                          csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
                                          )
        dataset_val = PE90DotNoSupportDatasetGroupDRO(image_size=args.image_size,
                                        data_dir=args.data_dir,
                                        partition='val',
                                        task=args.task, # 'classification' or 'detection
                                        random_crop=False,
                                        random_flip=False,
                                        biased=args.biased,
                                        )
    else:
        dataset_train_real = MedicalDevicePEDatasetGroupDRO(image_size=args.image_size,
                                          data_dir=args.data_dir,
                                          partition='train',
                                          task=args.task, # 'classification' or 'detection
                                          random_crop=args.random_crop,
                                          random_flip=args.random_flip,
                                          biased=args.biased,
                                          rebalance=True,
                                          csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
                                          ratio=args.balance_ratio,
                                          sample=True,
                                          )
        dataset_val = MedicalDevicePEDatasetGroupDRO(image_size=args.image_size,
                                        data_dir=args.data_dir,
                                        partition='val',
                                        task=args.task, # 'classification' or 'detection
                                        random_crop=False,
                                        random_flip=False,
                                        biased=args.biased,
                                        rebalance=True,
                                        csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
                                        ratio=0.1,
                                        )
        
    if args.augment:
        dataset_train_augmet = AugmentedMD(root_dir=args.augmented_data_dir, random_flip=args.random_flip, random_crop=args.random_crop, normalize=True, image_size=args.image_size)
        
        dataset_train = data.ConcatDataset([dataset_train_real, dataset_train_augmet])
        
    else:
        dataset_train = dataset_train_real
        
    print(f'Images on the dataset:{len(dataset_train)} [Train], {len(dataset_val)} [Val]', )

    train_dataloader = data.DataLoader(dataset_train, batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4, pin_memory=True)
    val_dataloader = data.DataLoader(dataset_val, batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=4, pin_memory=True)
    # Best AUROC value during training
    best_score = 0
    best_threshold = 0
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
        val_loss, threshold, auc_score = evaluating(epoch, model, val_dataloader, device, loss_criteria, mb)

        mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, auc_score))
        validation_losses.append(val_loss)
        validation_score.append(auc_score)

        # Update learning rate
        lr_scheduler.step(auc_score)

        # Update training chart
        mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0,epoch+1], [0,1])

        # Save model
        if best_score < auc_score:
            mb.write(f"Improve AUROC from {best_score} to {auc_score}")
            mb.write(f"best threshold so far: {threshold}")
            best_score = auc_score
            best_threshold = threshold
            nonimproved_epoch = 0
            torch.save({"model": model.state_dict(), 
                        "optimizer": optimizer.state_dict(), 
                        "best_score": best_score,
                        "best_threshold": best_threshold,
                        "epoch": epoch, 
                        "lr_scheduler": lr_scheduler.state_dict()}, args.model_path)
        else: 
            nonimproved_epoch += 1
        if nonimproved_epoch > 10:
            mb.write(f"Early stopping at epoch {epoch}")
            break    
    mb.write(f"Best AUROC: {best_score}")
    mb.write(f"Best Threshold: {best_threshold}")