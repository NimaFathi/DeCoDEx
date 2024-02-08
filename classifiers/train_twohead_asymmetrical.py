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

from core.image_datasets import MedicalDevicePEDataset, MedicalDeviceDataset
from torch.nn import functional as F
from itertools import zip_longest
from fastprogress import master_bar, progress_bar

class TwoHeadedDenseNet(nn.Module):
    def __init__(self, num_classes=1):
        super(TwoHeadedDenseNet, self).__init__()
        self.backbone = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
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


def epoch_training(epoch, model, train_dataloader_cls, train_dataloader_det, device, loss_criteria, optimizer, mb):
    model.train()
    training_loss = 0

    # Create progress bars for both cls and det dataloaders
    cls_bar = progress_bar(train_dataloader_cls, parent=mb)
    det_bar = progress_bar(train_dataloader_det, parent=mb)
    weitht_cls = 10
    weitht_det = 1
    batch = 0
    for cls_data, det_data in zip_longest(cls_bar, det_bar, fillvalue=None):
        optimizer.zero_grad()

        # Process classifier data if available
        if cls_data is not None:
            batch += 1
            images_cls, labels_cls = cls_data
            images_cls, labels_cls = images_cls.to(device), labels_cls.to(device).float()
            cls_pred = model(images_cls, 0).squeeze(1)
            cls_loss = weitht_cls * loss_criteria(cls_pred, labels_cls)
            cls_loss.backward()
            training_loss += cls_loss.item()

        # Process detector data if available
        if det_data is not None:
            batch += 1
            images_det, labels_det = det_data
            images_det, labels_det = images_det.to(device), labels_det.to(device).float()
            det_pred = model(images_det, 1).squeeze(1)
            det_loss = weitht_det * loss_criteria(det_pred, labels_det)
            det_loss.backward()
            training_loss += det_loss.item()

        optimizer.step()
        mb.child.comment = f'Epoch {epoch}: Training loss {training_loss / batch}'

    return training_loss / max(len(train_dataloader_cls), len(train_dataloader_det))

from sklearn.metrics import roc_curve

def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate """
    fpr, tpr, threshold = roc_curve(target, predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold

def evaluating(epoch, model, val_loader, device, loss_criteria, mb):
    model.eval()

    val_loss = 0
    out_pred_cls = torch.FloatTensor().to(device)
    out_gt_cls = torch.FloatTensor().to(device)
    out_pred_det = torch.FloatTensor().to(device)
    out_gt_det = torch.FloatTensor().to(device)

    with torch.no_grad():
        for step, (images, labels) in enumerate(progress_bar(val_loader, parent=mb)):
            images = images.to(device)
            cls_labels = labels['y'].to(device).float()
            det_labels = labels['z'].to(device).float()

            out_gt_cls = torch.cat((out_gt_cls, cls_labels), 0)
            out_gt_det = torch.cat((out_gt_det, det_labels), 0)

            cls_ps = model(images, 0).squeeze(1)
            det_ps = model(images, 1).squeeze(1)

            out_pred_cls = torch.cat((out_pred_cls, cls_ps), 0)
            out_pred_det = torch.cat((out_pred_det, det_ps), 0)

            cls_loss = loss_criteria(cls_ps, cls_labels)
            det_loss = loss_criteria(det_ps, det_labels)

            val_loss += cls_loss.item() + det_loss.item()

    cls_threshold = find_optimal_cutoff(out_gt_cls.to("cpu").numpy(), out_pred_cls.to("cpu").numpy())
    det_threshold = find_optimal_cutoff(out_gt_det.to("cpu").numpy(), out_pred_det.to("cpu").numpy())

    # Convert continuous predictions to binary predictions using the thresholds
    cls_binary_predictions = (out_pred_cls > cls_threshold).float()
    det_binary_predictions = (out_pred_det > det_threshold).float()

    # Calculate ROC AUC scores based on binary predictions
    cls_roc_auc = roc_auc_score(out_gt_cls.to("cpu").numpy(), cls_binary_predictions.to("cpu").numpy())
    det_roc_auc = roc_auc_score(out_gt_det.to("cpu").numpy(), det_binary_predictions.to("cpu").numpy())

    del images, labels, cls_loss, det_loss, cls_ps, det_ps
    if torch.cuda.is_available(): 
        torch.cuda.empty_cache()

    return (val_loss/len(val_loader),
            cls_threshold,
            det_threshold,
            cls_roc_auc,
            det_roc_auc)


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
    parser.add_argument("--task", type=str, default="both", choices=["both"],
                        help="Task for the model")
    parser.add_argument("--random_crop", action="store_false", help="Use random cropping as augmentation")
    parser.add_argument("--random_flip", action="store_false", help="Use random flipping as augmentation")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--balanced", action="store_false", help="Use balanced dataset")
    args = parser.parse_args()
    args.model_path = f"pretrained/md_{args.task}_densenet121_{args.image_size}_logits_twohead_imbalanced_correct2.pth"
    print(args.balanced)
    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")


    #define a model 
    # usind num_classes = 1 for binary classification
    model = TwoHeadedDenseNet(num_classes=1)
    model = model.to(device)

    #define loss function
    # Loss function
    loss_criteria = nn.BCEWithLogitsLoss()

    # Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    # Learning rate will be reduced automatically during training
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = args.lr_sf, patience = args.lr_patience, mode = 'min', verbose=True)
    dataset_train_cls = MedicalDevicePEDataset(image_size=args.image_size,
                                          data_dir=args.data_dir,
                                          partition='train',
                                          task="classification", # 'classification' or 'detection
                                          random_crop=args.random_crop,
                                          random_flip=args.random_flip,
                                          csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
                                          rebalance=args.balanced,
                                          ratio=0.1
                                          )
    
    dataset_train_det = MedicalDeviceDataset(image_size=args.image_size,
                                          data_dir=args.data_dir,
                                          partition='train',
                                          task='detection', # 'classification' or 'detection
                                          random_crop=args.random_crop,
                                          random_flip=args.random_flip,
                                          csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
                                          )
    
    dataset_val = MedicalDevicePEDataset(image_size=args.image_size,
                                        data_dir=args.data_dir,
                                        partition='val',
                                        task="both", # 'classification' or 'detection
                                        random_crop=False,
                                        random_flip=False,
                                        csv_dir='/usr/local/data/nimafh/midl2024-cfdiffusion/datasets',
                                        rebalance=args.balanced,
                                        ratio=0.1
                                        )
    
    print(f'Images on the dataset:{len(dataset_train_cls) + len(dataset_train_det)} [Train], {len(dataset_val)} [Val]', )

    train_dataloader_cls = data.DataLoader(dataset_train_cls, batch_size=2*args.batch_size,
                                shuffle=True,
                                num_workers=4, pin_memory=True)
    train_dataloader_det = data.DataLoader(dataset_train_det, batch_size=args.batch_size,
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
        train_loss = epoch_training(epoch, model, train_dataloader_cls, train_dataloader_det, device, loss_criteria, optimizer, mb)
        mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))
        training_losses.append(train_loss)

        # Evaluating
        val_loss, cls_threshold, det_thresholds, new_score_cls, new_score_det = evaluating(epoch, model, val_dataloader, device, loss_criteria, mb)
        
        mb.write('Finish validation epoch {} with loss {:.4f} and score csl {:.4f} and score det {:.4f}'.format(epoch, val_loss, new_score_cls, new_score_det))
        validation_losses.append(val_loss)
        validation_score.append((new_score_cls, new_score_det))
        # Update learning rate
        lr_scheduler.step(val_loss)
        avg_score = (new_score_cls + new_score_det) / 2
        # Update training chart
        mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0,epoch+1], [0,1])

        # Save model
        if best_score < avg_score:
            mb.write(f"Improve AUROC from {best_score} to {avg_score}")
            mb.write(f"classifier threshold: {cls_threshold}, detector threshold: {det_thresholds}")
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