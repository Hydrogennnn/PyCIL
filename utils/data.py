import numpy as np
from numpy.random.mtrand import f
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from . import autoaugment
from . import ops
from sklearn.model_selection import train_test_split
import os
from .transforms import ArrayToTensor, DataStack, GroupNormalize, IdentityTransform, ImgStack, ToTorchFormatTensor, GroupScale, GroupCenterCrop
import torch
from backbones.TBN import TBN
from collections import OrderedDict


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100_AA(iCIFAR100):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iCIFAR10_AA(iCIFAR10):
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        autoaugment.CIFAR10Policy(),
        transforms.ToTensor(),
        ops.Cutout(n_holes=1, length=16),
    ]


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class AVE(iData):
    class_order = [18, 16, 10, 21, 27, 6, 5, 11, 14, 9, 24, 12, 19, 15, 25, 1, 13, 2, 17, 26, 4, 22, 7, 8, 23, 0, 3, 20]
    # class_order = [17, 9, 13, 22, 27, 2, 5, 11, 18, 14, 25, 16, 4, 6, 0, 3, 21, 26, 15, 19, 8, 10, 23, 7, 1, 24, 12, 20]
    
    train_trsf = [transforms.ToTensor()]
    test_trsf = [transforms.ToTensor()]

    def download_data(self):
        # split = np.load("data/AVE_features/split.npy", allow_pickle=True).item()
        # self.train_data_idx = split["train"]
        # self.test_data_idx = split["test"]
        # self.val_data_idx = split["val"]

        # self.all_targets = np.load("data/AVE_features/labels.npy", allow_pickle=True)
        # self.train_targets = self.all_targets[self.train_data_idx]
        # self.test_targets = self.all_targets[self.test_data_idx]
        # self.val_targets = self.all_targets[self.val_data_idx]

        # self.train_data = self.train_data_idx
        # self.test_data = self.test_data_idx
        # self.val_data = self.val_data_idx

        self.split = np.load("data/AVE_features/split.npy", allow_pickle=True).item()
        self.all_targets = np.load("data/AVE_features/labels.npy", allow_pickle=True).item()

        self.train_data = self.split["train"]
        self.test_data = self.split["test"]
        self.val_data = self.split["val"]

        self.train_targets = self.all_targets["train"]
        self.test_targets = self.all_targets["test"]
        self.val_targets = self.all_targets["val"]



class Kinetics(iData):
    class_order = [23, 16, 1, 5, 6, 7, 11, 26, 21, 28, 0, 24, 3, 10, 8, 14, 15, 19, 17, 25, 9, 12, 2, 22, 20, 27, 18, 29, 13, 4]

    train_trsf = [transforms.ToTensor()]
    test_trsf = [transforms.ToTensor()]

    def download_data(self):
        self.split = np.load("data/KS/split.npy", allow_pickle=True).item()
        self.all_targets = np.load("data/KS/labels.npy", allow_pickle=True).item()

        self.train_data = self.split["train"]
        self.test_data = self.split["test"]
        self.val_data = self.split["val"]

        self.train_targets = self.all_targets["train"]
        self.test_targets = self.all_targets["test"]
        self.val_targets = self.all_targets["val"]
        
        



class VGGSound(iData):
    class_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 
                   17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                   32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 
                   47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
                   62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 
                   77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 
                   92, 93, 94, 95, 96, 97, 98, 99]
    train_trsf = [transforms.ToTensor()]
    test_trsf = [transforms.ToTensor()]

    def download_data(self):
        self.split = np.load("data/VGG/split.npy", allow_pickle=True).item()
        self.all_targets = np.load("data/VGG/labels.npy", allow_pickle=True).item()

        self.train_data = self.split["train"]
        self.test_data = self.split["test"]
        self.val_data = self.split["val"]

        self.train_targets = self.all_targets["train"]
        self.test_targets = self.all_targets["test"]
        self.val_targets = self.all_targets["val"]

class MMEA_CL(iData):
    class_order = [
        26,
        14,
        23,
        4,
        11,
        25,
        31,
        10,
        29,
        5,
        6,
        9,
        17,
        22,
        2,
        19,
        13,
        1,
        21,
        16,
        8,
        3,
        27,
        28,
        15,
        30,
        0,
        7,
        12,
        18,
        20,
        24
    ]


    def __init__(self):
        self.modality = ["RGB", "Gyro", "Acce"]
        self.arch = "BNInception"
        # self.train_list = train_list
        # self.test_list = test_list
        dataroot = "data/UESTC-MMEA-CL/"
        self.train_list = os.path.join(dataroot, "mydataset_train.txt")
        self.test_list = os.path.join(dataroot, "mydataset_test.txt")
        self.val_list = os.path.join(dataroot, "mydataset_val.txt")



        new_length = OrderedDict({
                        ('RGB', 1),
                        ('Gyro', 24),
                        ('Acce', 24)
                    })
        model = TBN(num_segments=8, modality=["RGB", "Gyro", "Acce"],
        base_model='BNInception', new_length=new_length)

        self.crop_size = model.crop_size
        self.scale_size = model.scale_size
        self.input_mean = model.input_mean
        self.input_std = model.input_std
        self.data_length = model.new_length
        self.train_augmentation = model.get_augmentation()

        self.train_trsf = {}
        self.test_trsf = {}
        self.normalize = {}


        
    def download_data(self):
        
        for m in self.modality:
            if (m != 'RGBDiff'):
                self.normalize[m] = GroupNormalize(self.input_mean[m], self.input_std[m])
            else:
                self.normalize[m] = IdentityTransform()

        for m in self.modality:
            if (m != 'Gyro' and m != 'Acce'):
                # Prepare train/val dictionaries containing the transformations
                # (augmentation+normalization)
                # for each modality
                self.train_trsf[m] = transforms.Compose([
                self.train_augmentation[m],
                ImgStack(roll=self.arch == 'BNInception'),
                ToTorchFormatTensor(div=self.arch != 'BNInception'),
                self.normalize[m],
                ])

                self.test_trsf[m] = transforms.Compose([
                    GroupScale(int(self.scale_size[m])),
                    GroupCenterCrop(self.crop_size[m]),
                    ImgStack(roll=self.arch == 'BNInception'),
                    ToTorchFormatTensor(div=self.arch != 'BNInception'),
                    self.normalize[m],
                ])
            else:
                self.train_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                    self.normalize[m],
                ])

                self.test_trsf[m] = transforms.Compose([
                    DataStack(),
                    ArrayToTensor(),
                    self.normalize[m],
                ])
            

        val_set = MMEADataSet(self.val_list)
        train_set = MMEADataSet(self.train_list)
        test_set = MMEADataSet(self.test_list)

        self.train_data, self.test_data, self.val_data = np.array(train_set.video_list), np.array(test_set.video_list), np.array(val_set.video_list)
        self.train_targets, self.test_targets, self.val_targets = np.array(self._get_targets(train_set)), np.array(self._get_targets(test_set)), np.array(self._get_targets(val_set))
        

    def _get_targets(self, dataset):
        """
        get target list from MyDataset
        """
        targets = []
        for i in range(len(dataset)):
            targets.append(dataset.video_list[i].label)

        return targets

class MMEADataSet(torch.utils.data.Dataset):
    def __init__(self, list_file):
        self.list_file = list_file

        

        class MyDataset_VideoRecord():

            def __init__(self, row):
                self.data = row

            @property
            def path(self):
                return self.data[0]

            @property
            def num_frames(self):
                return {'RGB': int(self.data[1]),
                        'Flow': int(self.data[1])-1,
                        'Gyro': int(self.data[2]),
                        'Acce': int(self.data[2])}

            @property
            def label(self):
                return int(self.data[3])

        self.MyDataset_VideoRecord = MyDataset_VideoRecord

        self._parse_list()

    def _parse_list(self):
        
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [self.MyDataset_VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def __len__(self):
        return len(self.video_list)