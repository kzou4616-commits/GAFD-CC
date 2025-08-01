import time
import torch
import torch.nn as nn
import argparse
import os
from tqdm import tqdm
import mgzip
import pickle




def get_features(model, data_loader, args,device):
    save_path = os.path.join(args.cache_dir, args.model, args.ind_dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logits_file_path = os.path.join(save_path, "logits.pkl")
    features_file_path = os.path.join(save_path, "features.pkl")
    labels_file_path = os.path.join(save_path, "labels.pkl")  # 新增 labels 的缓存文件路径
    if os.path.exists(logits_file_path) and os.path.exists(features_file_path) and os.path.exists(labels_file_path) and args.use_feature_cache:
        with mgzip.open(logits_file_path, "rb",thread=args.num_workers) as f:
            save_logits = pickle.load(f)
        with mgzip.open(features_file_path, "rb",thread=args.num_workers) as f:
            save_features = pickle.load(f)
        with mgzip.open(labels_file_path, "rb", thread=args.num_workers) as f:
            save_labels = pickle.load(f)
        print("read")
    else:

        model.eval()
        features = [[] for i in range(args.num_classes)]
        logits = [[] for i in range(args.num_classes)]
        labels_list = [[] for i in range(args.num_classes)]  # 新增 labels 收集列表
        with torch.no_grad():
            print(data_loader)
            for (images, labels) in tqdm(data_loader):
                images, labels = images.to(device), labels.to(device)
                output, feature = model.get_feature(images)
                
                p_labels = output.argmax(1)
                for i in range(labels.size(0)):
                    logits[p_labels[i]].append(output[i].cpu())
                    features[p_labels[i]].append(feature[i].cpu())
                    labels_list[p_labels[i]].append(labels[i].cpu())  # 收集 labels

        save_features = []
        save_logits = []
        save_labels = []  # 新增保存 labels 的列表
        for i in range(args.num_classes):
            if len(logits[i])==0:
                save_features.append(torch.Tensor([]))
                save_logits.append(torch.Tensor([]))
                save_labels.append(torch.Tensor([]))  # 将空 tensor 添加到 save_labels
                continue
            tmp = torch.stack(features[i], dim=0)
            save_features.append(tmp)
            tmp = torch.stack(logits[i], dim=0)
            save_logits.append(tmp)
            tmp = torch.stack(labels_list[i], dim=0)  # 将当前类别的 labels 转换为 tensor
            save_labels.append(tmp)
        with mgzip.open(logits_file_path, "wb",thread=args.num_workers) as f:
            pickle.dump(save_logits, f)
        with mgzip.open(features_file_path, "wb",thread=args.num_workers) as f:
            pickle.dump(save_features, f)
        with mgzip.open(labels_file_path, "wb", thread=args.num_workers) as f:
            pickle.dump(save_labels, f)
        print("write")

    return save_features, save_logits, save_labels  # 返回 features, logits, labels


