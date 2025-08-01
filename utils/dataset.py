from torchvision import datasets, transforms
# import os
from torch.utils.data import Subset
import torch.utils.data
def get_dataset(dataset,args):
    train_dataset = None
    test_dataset = None
    if  dataset == "ImageNet":
        size = 224
    else:
        size = 32
    #small-scale dataset
    transforme_mean = (0.4914, 0.4822, 0.4465)
    transforme_std =(0.2023, 0.1994, 0.2010)
    # ind dataset

    # small-scale dataset
    if dataset == "cifar10":
        from torchvision.datasets import CIFAR10
        train_transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = CIFAR10("/data0/zoukun/OODwork/data/cifar10", train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10("/data0/zoukun/OODwork/data/cifar10", train=False, transform=test_transform, download=True)

    elif dataset == "cifar100":
        from torchvision.datasets import CIFAR100
        train_transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = CIFAR100("/data0/zoukun/OODwork/data/cifar100", train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100("/data0/zoukun/OODwork/data/cifar100", train=False, transform=test_transform, download=True)
    
    # large-scale dataset
    elif dataset == "ImageNet":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ImageNet-1000/train', transform=transform_test_largescale)
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ImageNet-1000/val', transform=transform_test_largescale)
        
        

    # ood dataset

    # small-scale dataset
    elif dataset == "iSUN":
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ss_ood_data/iSUN', transform=transform)
    
    
    elif dataset == "svhn":
        from torchvision.datasets import SVHN
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        test_dataset = SVHN("/data0/zoukun/OODwork/data/ss_ood_data/svhn", split='test', transform=transform, download=True)
    
    elif dataset == "dtd":
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ss_ood_data/dtd/images', transform=transform)
    elif dataset == "places365":
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        full_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ss_ood_data/places365', transform=transform)
        test_dataset = Subset(full_dataset, range(10000))
    elif dataset == "LSUN_crop":
        transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ss_ood_data/LSUN', transform=transform)

    elif dataset == "LSUN_resize":
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ss_ood_data/LSUN_resize', transform=transform)

    elif dataset == "TinyImageNet_crop":
        crop_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/TinyImagenet-crop', transform=crop_transform)

    elif dataset == "TinyImageNet_resize":
        transform = transforms.Compose([
            transforms.Resize([size,size]), 
            transforms.ToTensor(),
            transforms.Normalize(transforme_mean, transforme_std)
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/TinyImagenet-resize', transform=transform)


    
    # large-scale dataset
    elif dataset == "iNat":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/iNaturalist', transform=transform_test_largescale)

    elif dataset == "SUN":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/SUN', transform=transform_test_largescale)
    elif dataset == "Places":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/Places', transform=transform_test_largescale)

    elif dataset == "Textures":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/dtd/images', transform=transform_test_largescale)
    elif dataset == "ninco":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/ninco', transform=transform_test_largescale)
    elif dataset == "ssb_hard":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/ssb_hard', transform=transform_test_largescale)
        # print("ssb_hard",len(test_dataset))
    elif dataset == "openimage_o":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_dataset = None
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/openimage_o', transform=transform_test_largescale)
    elif dataset == "imagenet_o":
        transform_test_largescale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_dataset = datasets.ImageFolder(root='/data0/zoukun/OODwork/data/ls_ood_data/imagenet-o', transform=transform_test_largescale)
    return train_dataset, test_dataset

