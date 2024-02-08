import os
import yaml
import math
import random
import argparse
import itertools
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from PIL import Image
from time import time
from os import path as osp
from multiprocessing import Pool

import torch
from torch.utils import data

from torchvision import transforms
from torchvision import datasets
import sys
project_root= '../'
sys.path.append(project_root)

from core import dist_util
from core.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    args_to_dict,
    add_dict_to_argparser,
)
from core.sample_utils import (
    get_DiME_iterative_sampling,
    clean_class_cond_fn,
    clean_detect_cond_fn,
    dist_cond_fn,
    ImageSaver,
    SlowSingleLabel,
    Normalizer,
    load_from_DDP_model,
    PerceptualLoss,
    X_T_Saver,
    Z_T_Saver,
    ChunkedDataset,
)
from core.image_datasets import PEDataset
from core.gaussian_diffusion import _extract_into_tensor
from core.classifier.densenet import ClassificationModel

import matplotlib
matplotlib.use('Agg')  # to disable display



# =======================================================
# =======================================================
# Functions
# =======================================================
# =======================================================


def create_args():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        gpu='0',
        num_batches=50,
        use_train=False,
        dataset='CelebA',

        # path args
        output_path='',
        classifier_path='models/classifier.pth',
        detector_path='models/detector.pth',
        oracle_path='models/oracle.pth',
        model_path="models/ddpm-celeba.pt",
        csv_dir="",
        data_dir="",
        exp_name='',

        # sampling args
        classifier_scales='8,10,15',
        detector_scales='5,10,15',
        seed=4,
        use_ddim=False,
        start_step=60,
        use_logits=False,
        l1_loss=0.0,
        l2_loss=0.0,
        l_perc=0.0,
        l_perc_layer=1,
        vgg_pretrained=True,
        use_sampling_on_x_t=True,
        sampling_scale=1.,  # use this flag to rescale the variance of the noise
        guided_iterations=9999999,  # set a high number to do all iteration in a guided way

        # evaluation args
        merge_and_eval=False,  # when all chunks have finished, run it with this flag

        # misc args
        num_chunks=1,
        chunk=0,
        save_x_t=False,
        save_z_t=False,
        save_images=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser.parse_args()


# =======================================================
# =======================================================
# Merge all chunks' information and compute the
# overall metrics
# =======================================================
# =======================================================


def mean(array):
    m = np.mean(array).item()
    return 0 if math.isnan(m) else m


def merge_and_compute_overall_metrics(args, device):

    def div(q, p):
        if p == 0:
            return 0
        return q / p

    print('Merging all results ...')

    # read all yaml files containing the info to add them together
    summary = {
        'class-cor': {'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'class-inc': {'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
                      'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0},
        'clean acc': 0,
        'cf acc': 0,
        'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0,
    }

    for chunk in range(args.num_chunks):
        yaml_path = osp.join(args.output_path, 'Results', args.exp_name,
                             f'chunk-{chunk}_num-chunks-{args.num_chunks}_summary.yaml')

        with open(yaml_path, 'r') as f:
            chunk_summary = yaml.load(f, Loader=yaml.FullLoader)

        summary['clean acc'] += chunk_summary['clean acc'] * chunk_summary['n']
        summary['cf acc'] += chunk_summary['cf acc'] * chunk_summary['n']
        
        summary['n'] += chunk_summary['n']

        summary['class-cor']['n'] += chunk_summary['class-cor']['n']
        summary['class-inc']['n'] += chunk_summary['class-inc']['n']

        summary['cf-cor']['n'] += chunk_summary['cf-cor']['n']
        summary['cf-inc']['n'] += chunk_summary['cf-inc']['n']

        summary['class-cor']['cf-cor']['n'] += chunk_summary['class-cor']['cf-cor']['n']
        summary['class-cor']['cf-inc']['n'] += chunk_summary['class-cor']['cf-inc']['n']
        summary['class-inc']['cf-cor']['n'] += chunk_summary['class-inc']['cf-cor']['n']
        summary['class-inc']['cf-inc']['n'] += chunk_summary['class-inc']['cf-inc']['n']


        for k in ['bkl', 'l_1', 'FVA', 'MNAC']:
            summary[k] += chunk_summary[k] * chunk_summary['n']

            summary['class-cor'][k] += chunk_summary['class-cor'][k] * chunk_summary['class-cor']['n']
            summary['class-inc'][k] += chunk_summary['class-inc'][k] * chunk_summary['class-inc']['n']

            summary['cf-cor'][k] += chunk_summary['cf-cor'][k] * chunk_summary['cf-cor']['n']
            summary['cf-inc'][k] += chunk_summary['cf-inc'][k] * chunk_summary['cf-inc']['n']

            summary['class-cor']['cf-cor'][k] += chunk_summary['class-cor']['cf-cor'][k] * chunk_summary['class-cor']['cf-cor']['n']
            summary['class-cor']['cf-inc'][k] += chunk_summary['class-cor']['cf-inc'][k] * chunk_summary['class-cor']['cf-inc']['n']
            summary['class-inc']['cf-cor'][k] += chunk_summary['class-inc']['cf-cor'][k] * chunk_summary['class-inc']['cf-cor']['n']
            summary['class-inc']['cf-inc'][k] += chunk_summary['class-inc']['cf-inc'][k] * chunk_summary['class-inc']['cf-inc']['n']

    for k in ['cf acc', 'clean acc']:
        summary[k] = div(summary[k], summary['n'])

    for k in ['bkl', 'l_1', 'FVA', 'MNAC']:
        summary[k] = div(summary[k], summary['n'])

        summary['class-cor'][k] = div(summary['class-cor'][k], summary['class-cor']['n'])
        summary['class-inc'][k] = div(summary['class-inc'][k], summary['class-inc']['n'])

        summary['cf-cor'][k] = div(summary['cf-cor'][k], summary['cf-cor']['n'])
        summary['cf-inc'][k] = div(summary['cf-inc'][k], summary['cf-inc']['n'])

        summary['class-cor']['cf-cor'][k] = div(summary['class-cor']['cf-cor'][k], summary['class-cor']['cf-cor']['n'])
        summary['class-cor']['cf-inc'][k] = div(summary['class-cor']['cf-inc'][k], summary['class-cor']['cf-inc']['n'])
        summary['class-inc']['cf-cor'][k] = div(summary['class-inc']['cf-cor'][k], summary['class-inc']['cf-cor']['n'])
        summary['class-inc']['cf-inc'][k] = div(summary['class-inc']['cf-inc'][k], summary['class-inc']['cf-inc']['n'])

    # summary is ready to save
    print('done')
    print('Acc on the set:', summary['clean acc'])
    print('CF Acc on the set:', summary['cf acc'])

    with open(osp.join(args.output_path, 'Results', args.exp_name, 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f)


# =======================================================
# =======================================================
# Main
# =======================================================
# =======================================================


def main():

    args = create_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.makedirs(osp.join(args.output_path, 'Results', args.exp_name),
                exist_ok=True)

    # ========================================
    # Evaluate all feature in case of 
    if args.merge_and_eval:
        merge_and_compute_overall_metrics(args, dist_util.dev())
        return  # finish the script

    # ========================================
    # Set seeds

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================
    # Load Dataset


    if args.dataset == 'PE_Normal':
        dataset = PEDataset(image_size=args.image_size,
                            task='both',
                            data_dir=args.data_dir,
                            partition='test',
                            random_crop=False,
                            random_flip=False,
                            )


    # breaks the dataset into chunks
    dataset = ChunkedDataset(dataset=dataset,
                             chunk=args.chunk,
                             num_chunks=args.num_chunks)

    print('Images on the dataset:', len(dataset))

    loader = data.DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True)

    # ========================================
    # load models

    print('Loading Model and diffusion model')
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    print('Loading Classifier')

    classifier = ClassificationModel(args.classifier_path).to(dist_util.dev())
    classifier.eval()
    detector = ClassificationModel(args.detector_path).to(dist_util.dev())
    detector.eval()
    # ========================================
    # Distance losses

    if args.l_perc != 0:
        print('Loading Perceptual Loss')
        vggloss = PerceptualLoss(layer=args.l_perc_layer,
                                 c=args.l_perc, pretrained=args.vgg_pretrained).to(dist_util.dev())
        vggloss.eval()
    else:
        vggloss = None

    # ========================================
    # get custom function for the forward phase
    # and other variables of interest

    sample_fn = get_DiME_iterative_sampling(use_sampling=args.use_sampling_on_x_t)

    x_t_saver = X_T_Saver(args.output_path, args.exp_name) if args.save_x_t else None
    z_t_saver = Z_T_Saver(args.output_path, args.exp_name) if args.save_z_t else None
    save_imgs = ImageSaver(args.output_path, args.exp_name, extention='.jpg') if args.save_images else None

    current_idx = 0
    start_time = time()

    stats = {
        'n': 0,
        'flipped': 0,
        'bkl': [],
        'l_1': [],
        'pred': [],
        'cf pred': [],
        'target': [],
        'label': [],
        'det_label': [],
    }

    acc = 0
    n = 0
    classifier_scales = [float(x) for x in args.classifier_scales.split(',')]
    detector_sacles = [float(x) for x in args.detector_scales.split(',')]

    print('Starting Image Generation')
    for idx, (indexes, img, lab) in enumerate(loader):
        print(f'[Chunk {args.chunk + 1} / {args.num_chunks}] {idx} / {min(args.num_batches, len(loader))} | Time: {int(time() - start_time)}s')

        img = img.to(dist_util.dev())
        I = (img / 2) + 0.5
        cls_lab = lab['y']
        det_lab = lab['z']
        cls_lab = cls_lab.to(dist_util.dev(), dtype=torch.long)
        det_lab = det_lab.to(dist_util.dev(), dtype=torch.long)
        t = torch.zeros(img.size(0), device=dist_util.dev(),
                        dtype=torch.long)

        # Initial Classification, no noise included
        with torch.no_grad():
            logits = classifier(img)
            # logits are between -inf and +inf so we can use a threshold of 0.0
            # TODO: check if sigmoid is applied in the classifier
            pred = (logits > 0.0).long()
            md_logits = detector(img)
            md_pred = (md_logits > 0.0).long()

        acc += (pred == cls_lab).sum().item()
        n += cls_lab.size(0)

        # as the model is binary, the target will always be the inverse of the prediction
        target = 1 - pred
        target_det = md_logits
        target_det_bool = md_pred

        t = torch.ones_like(t) * args.start_step

        # add noise to the input image 
        noise_img = diffusion.q_sample(img, t)

        transformed = torch.zeros_like(cls_lab).bool()
        transformed_det = torch.zeros_like(det_lab).bool()

        ccfd = 0
        flag=False
        for jdx, classifier_scale in enumerate(classifier_scales):
            if flag:
                break
            for jjdx, detector_scale in enumerate(detector_sacles):
                # choose the target label
                model_kwargs = {}
                model_kwargs['y'] = target[~transformed]
                # sample image from the noisy_img
                cfs, xs_t_s, zs_t_s = sample_fn(
                    diffusion,
                    model_fn,
                    img[~transformed, ...].shape,
                    args.start_step,
                    img[~transformed, ...],
                    t,
                    z_t=noise_img[~transformed, ...],
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    device=dist_util.dev(),
                    class_grad_fn=clean_class_cond_fn,
                    class_grad_kwargs={'y': target[~transformed],
                                    'classifier': classifier,
                                    's': classifier_scale,
                                    'use_logits': args.use_logits},
                    dist_grad_fn=dist_cond_fn,
                    dist_grad_kargs={'l1_loss': args.l1_loss,
                                    'l2_loss': args.l2_loss,
                                    'l_perc': vggloss},
                    detect_grad_fn=clean_detect_cond_fn,
                    detect_grad_kargs={'y': target_det[~transformed],
                                        'detector': detector,
                                        's': detector_scale,
                                        'use_logits': args.use_logits},
                    guided_iterations=args.guided_iterations,
                    is_x_t_sampling=False
                )

                # evaluate the cf and check whether the model flipped the prediction
                with torch.no_grad():
                    cfsl = classifier(cfs)
                    cfsp = (cfsl > 0.0).bool()

                    cfdl = detector(cfs)
                    cfdp = (cfdl > 0.0).bool()
                
                if jdx == 0 and jjdx == 0:
                    cf = cfs.clone().detach()
                    x_t_s = [xp.clone().detach() for xp in xs_t_s]
                    z_t_s = [zp.clone().detach() for zp in zs_t_s]

                cf[~transformed] = cfs
                for kdx in range(len(x_t_s)):
                    x_t_s[kdx][~transformed] = xs_t_s[kdx]
                    z_t_s[kdx][~transformed] = zs_t_s[kdx]
                
                cfsp = cfsp.squeeze()
                cfdp = cfdp.squeeze()
                target = target.squeeze()
                target_det_bool = target_det_bool.squeeze()
                transformed_det[~transformed] = target_det_bool[~transformed] == cfdp

                transformed[~transformed] = target[~transformed] == cfsp

                # check if detection is correct for all samples break else continue
                if transformed.float().sum().item() == transformed.size(0):
                    if transformed_det.float().sum().item() == transformed_det.size(0):
                        final_cf = cf
                        ccfd = 1
                        flag=True
                        break
                    else:
                        final_cf = cf
                        ccfd = 1
                        # we reset transformed to all false
                        transformed = torch.zeros_like(cls_lab).bool()
                
        if ccfd == 0:
            final_cf = cf

        if args.save_x_t:
            x_t_saver(x_t_s, indexes=indexes)

        if args.save_z_t:
            z_t_saver(z_t_s, indexes=indexes)

        with torch.no_grad():
            logits_cf = classifier(final_cf)
            pred_cf = (logits_cf > 0.0).long()

            logits_det = detector(final_cf)
            pred_det = (logits_det > 0.0).long() 

            # adjusting the dimensions
            pred_cf = pred_cf.squeeze()
            pred_det = pred_det.squeeze()
            pred = pred.squeeze()
            
            # process images
            final_cf = ((final_cf + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            final_cf = final_cf.permute(0, 2, 3, 1)
            final_cf = final_cf.contiguous().cpu()

            I = (I * 255).to(torch.uint8)
            I = I.permute(0, 2, 3, 1)
            I = I.contiguous().cpu()

            noise_img = ((noise_img + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            noise_img = noise_img.permute(0, 2, 3, 1)
            noise_img = noise_img.contiguous().cpu()

            # add metrics
            dist_cf = torch.sigmoid(logits_cf)
            dist_cf[target == 0] = 1 - dist_cf[target == 0]
            bkl = (1 - dist_cf).detach().cpu()

            # dists
            I_f = (I.to(dtype=torch.float) / 255).view(I.size(0), -1)
            cf_f = (final_cf.to(dtype=torch.float) / 255).view(I.size(0), -1)
            l_1 = (I_f - cf_f).abs().mean(dim=1).detach().cpu()

            stats['l_1'].append(l_1)
            stats['n'] += I.size(0)
            stats['bkl'].append(bkl)
            stats['flipped'] += (pred_cf == target).sum().item()
            stats['cf pred'].append(pred_cf.detach().cpu())
            stats['target'].append(target.detach().cpu())
            stats['label'].append(cls_lab.detach().cpu())
            stats['pred'].append(pred.detach().cpu())
            stats['det_label'].append(det_lab.detach().cpu())
            print("Keys of stats",stats.keys())
            print('Stats of cf pred',stats['cf pred'])
            print('Stats of target',stats['target'])
            print('Stats of label',stats['label'])
            print('Stats of pred',stats['pred'])
            print('Stats of flipped',stats['flipped'])
            
        if args.save_images:
            save_imgs(imgs=I.numpy(), cfs=final_cf.numpy(), noises=noise_img.numpy(),
                        bkl=bkl.numpy(), l_1=l_1, indexes=indexes.numpy(), cls_lab=cls_lab,
                        c_pred=pred, cf_pred=pred_cf, cls_target=target, det_lab=det_lab,
                        det_pred=pred_det, det_target=target_det_bool)


        if (idx + 1) == min(args.num_batches, len(loader)):
            print(f'[Chunk {args.chunk + 1} / {args.num_chunks}] {idx + 1} / {min(args.num_batches, len(loader))} | Time: {int(time() - start_time)}s')
            print('\nDone')
            break

        current_idx += I.size(0)

    # write summary for all four combinations
    summary = {
        'class-cor': {'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0},
                      'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0},
                      'bkl': 0, 'l_1': 0, 'n': 0},
        'class-inc': {'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0},
                      'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0},
                      'bkl': 0, 'l_1': 0, 'n': 0},
        'cf-cor': {'bkl': 0, 'l_1': 0, 'n': 0},
        'cf-inc': {'bkl': 0, 'l_1': 0, 'n': 0},
        'clean acc': 100 * acc / n,
        'cf acc': stats['flipped'] / n,
        'bkl': 0, 'l_1': 0, 'n': 0, 'FVA': 0, 'MNAC': 0,
    }

    for k in stats.keys():
        if k in ['flipped', 'n']:
            continue
        stats[k] = torch.cat(stats[k]).numpy()

    for k in ['bkl', 'l_1']:

        summary['class-cor']['cf-cor'][k] = mean(stats[k][(stats['label'] == stats['pred']) & (stats['target'] == stats['cf pred'])])
        summary['class-inc']['cf-cor'][k] = mean(stats[k][(stats['label'] != stats['pred']) & (stats['target'] == stats['cf pred'])])
        summary['class-cor']['cf-inc'][k] = mean(stats[k][(stats['label'] == stats['pred']) & (stats['target'] != stats['cf pred'])])
        summary['class-inc']['cf-inc'][k] = mean(stats[k][(stats['label'] != stats['pred']) & (stats['target'] != stats['cf pred'])])

        summary['class-cor'][k] = mean(stats[k][stats['label'] == stats['pred']])
        summary['class-inc'][k] = mean(stats[k][stats['label'] != stats['pred']])

        summary['cf-cor'][k] = mean(stats[k][stats['target'] == stats['cf pred']])
        summary['cf-inc'][k] = mean(stats[k][stats['target'] != stats['cf pred']])

        summary[k] = mean(stats[k])

    summary['class-cor']['cf-cor']['n'] = len(stats[k][(stats['label'] == stats['pred']) & (stats['target'] == stats['cf pred'])])
    summary['class-inc']['cf-cor']['n'] = len(stats[k][(stats['label'] != stats['pred']) & (stats['target'] == stats['cf pred'])])
    summary['class-cor']['cf-inc']['n'] = len(stats[k][(stats['label'] == stats['pred']) & (stats['target'] != stats['cf pred'])])
    summary['class-inc']['cf-inc']['n'] = len(stats[k][(stats['label'] != stats['pred']) & (stats['target'] != stats['cf pred'])])

    summary['class-cor']['n'] = len(stats[k][stats['label'] == stats['pred']])
    summary['class-inc']['n'] = len(stats[k][stats['label'] != stats['pred']])

    summary['cf-cor']['n'] = len(stats[k][stats['target'] == stats['cf pred']])
    summary['cf-inc']['n'] = len(stats[k][stats['target'] != stats['cf pred']])

    summary['n'] = n

    print('ACC ON THIS SET:', 100 * acc / n)
    stats['acc'] = 100 * acc / n

    prefix = f'chunk-{args.chunk}_num-chunks-{args.num_chunks}_' if args.num_chunks != 1 else ''
    torch.save(stats, osp.join(args.output_path, 'Results', args.exp_name, prefix + 'stats.pth'))

    # save summary
    with open(osp.join(args.output_path, 'Results', args.exp_name, prefix + 'summary.yaml'), 'w') as f:
        yaml.dump(summary, f)


if __name__ == '__main__':
    main()
