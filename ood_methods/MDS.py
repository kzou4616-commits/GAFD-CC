import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class Mahalanobis:
    def __init__(self, model, args, device):
        self.model = model
        self.device = device
        self.num_classes = args.num_classes
        self.train_mean = None
        self.precision = None  # 协方差矩阵的逆

    def set_state(self, train_mean, precision):
        self.train_mean = train_mean
        self.precision = precision

    @torch.no_grad()
    def get_state(self, features, labels):
        # 计算每个类别的特征均值和协方差矩阵
        # features 是一个列表，每个元素是一个类别对应的特征张量
        # labels 是一个列表，每个元素是一个类别对应的标签张量    
    
    # 合并所有类别的特征
        all_features = torch.cat([feat for feat in features if feat.numel() > 0], dim=0)
        all_labels = torch.cat([lbl for lbl in labels if lbl.numel() > 0], dim=0)

        class_means = []
        centered_data = []
        for c in range(self.num_classes):
            class_features = all_features[all_labels == c]
            class_mean = class_features.mean(dim=0)
            class_means.append(class_mean)
            centered_data.append(class_features - class_mean.view(1, -1))

        class_means = torch.stack(class_means).to(self.device)

        # 计算协方差矩阵
        cov_matrix = torch.zeros(all_features.size(1), all_features.size(1)).to(self.device)
        for c in range(self.num_classes):
            diff = centered_data[c].to(self.device)
            cov_matrix += diff.t().mm(diff)
        cov_matrix = cov_matrix / all_features.size(0)
        precision = torch.inverse(cov_matrix + 0.0005 * torch.eye(cov_matrix.size(0)).to(self.device))
    
        return class_means, precision
    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            logit, feature = self.model.get_feature(images)
            class_ids = torch.argmax(torch.softmax(logit, dim=1), dim=1).cpu().numpy()
            tm = self.train_mean[class_ids].to(self.device)
            diff = feature - tm
            mahalanobis_distance = torch.sum(diff.mm(self.precision) * diff, dim=1)
            result.append(-mahalanobis_distance.cpu().numpy())  # 负号表示越小的距离得分越高
        return np.concatenate(result)