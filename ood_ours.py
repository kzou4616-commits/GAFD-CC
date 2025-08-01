import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.dataset import get_dataset
from utils.models import get_model
from utils.utils import fix_random_seed
from utils.metrics import cal_metric
from utils.get_stat import get_features

from ood_methods.ours import GADF

def ours_method():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ind_dataset", type=str, default="ImageNet", help="in-distribution dataset name,cifar100")
    parser.add_argument("--ood_dataset", type=str, default="iSUN", help="ood dataset ")
    parser.add_argument("--model", type=str, default="vit", help="vit or densenet ,resnet,vit,convnext,densenet,regnet,efficientnet,swin")
    parser.add_argument("--gpu", type=int, default=6, help="gpu id")
    parser.add_argument("--bs", type=int, default=32, help="batch size")
    parser.add_argument("--random_seed", type=int, default=0,help="random seed")
    parser.add_argument("--use_feature_cache", type=bool, default=True, help="use feature cache")
    parser.add_argument("--use_score_cache", type=bool, default=True, help="use score cache")
    parser.add_argument("--cache_dir", type=str, default="cache",help="cache directory")
    parser.add_argument("--result_dir", type=str, default="result",help="result directory")
    parser.add_argument("--plot_save_dir", type=str, default="./score_plots/",help="plot save directory")
    parser.add_argument("--num_workers", type=int, default=24,help="number of workers")
    parser.add_argument("--OOD_method", type=str, default="gafdcc",help="OOD method name",choices=["MSP","ODIN","Energy","GEN","ReAct","DICE","GradNorm","MaxLogit","ASH","OptFS","VIM","Residual","CARef","CADRef"])
    parser.add_argument("--logit_method", type=str, default="Energy",choices=["Energy","MSP","MaxLogit","GEN"],help="logit method for CADRef")
    args = parser.parse_args()

    fix_random_seed(args.random_seed)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.ind_dataset == "ImageNet":
        args.num_classes = 1000
    elif args.ind_dataset == "cifar10":
        args.num_classes = 10
    elif args.ind_dataset == "cifar100":
        args.num_classes = 100
    print(args)

    model = get_model(args).to(device)
    model.eval()

    _, ind_data = get_dataset(args.ind_dataset, args)
    ind_loader = DataLoader(dataset=ind_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)

    my_evaluator = GADF(model, args, device)
   
    train_data, _ = get_dataset(args.ind_dataset,args)
    train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
    features, logits, _ = get_features(model, train_loader, args, device)
    train_mean, global_mean_logit_score, class_mean_logit_score = my_evaluator.get_state(features, logits)
    my_evaluator.set_state(train_mean, global_mean_logit_score, class_mean_logit_score)

    ind_scores, ood_scores = None, None

    ind_scores = my_evaluator.eval(ind_loader)
    ind_labels = np.ones(ind_scores.shape[0])
    print(f"In-distribution dataset size: {len(ind_scores)}")
    # ["iNat","SUN","Places","Textures","openimage_o","imagenet_o","ssb_hard","ninco"]
    # ["iSUN","svhn","dtd","places365","LSUN_crop","LSUN_resize"]
    all_auroc = []
    all_fpr = []
    for ood_dataset in ["iSUN","svhn","dtd","places365","LSUN_crop","LSUN_resize"]:
        _, ood_data = get_dataset(ood_dataset,args)
        ood_loader = DataLoader(dataset=ood_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
        ood_scores = my_evaluator.eval(ood_loader)
        ood_labels = np.zeros(ood_scores.shape[0])
        scores = np.concatenate([ind_scores, ood_scores])
        labels = np.concatenate([ind_labels, ood_labels])
        auroc, aupr, fpr = cal_metric(labels, scores)
        auroc = round(auroc*100, 4)
        fpr = round(fpr*100, 4)
        all_auroc.append(auroc)
        all_fpr.append(fpr)
        print(f"{ood_dataset} dataset size: {len(ood_scores)}")
        print("{:10} {:10} {:10} {:10}".format(args.OOD_method, ood_dataset, auroc, fpr))
        with open(os.path.join(args.result_dir, args.model,args.ind_dataset, args.OOD_method+".txt"), "a") as f:
            f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ood_dataset, auroc, fpr))
    # average
    average_auroc = np.mean(all_auroc)
    average_fpr = np.mean(all_fpr)
    print(f"Average auroc: {average_auroc:.4f}")
    print(f"Average fpr: {average_fpr:.4f}")

if __name__ == '__main__':
    ours_method()