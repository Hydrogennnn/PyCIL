import logging
import numpy as np
from numpy.lib.arraysetops import isin
from PIL import Image
from torch.nn.functional import instance_norm
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR10_AA, iCIFAR100_AA, AVE
from tqdm import tqdm
import torch

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, aug=1):
        self.dataset_name = dataset_name
        self.aug = aug
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]
    
    def get_accumulate_tasksize(self,task):
        return sum(self._increments[:task+1])
    
    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x_idxs, y = self._train_data_idx, self._train_targets
        elif source == "test":
            x_idxs, y = self._test_data_idx, self._test_targets
        elif source == "val":
            x_idxs, y = self._val_data_idx, self._val_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test" or mode == "val":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # data, targets = [], []
        data_indices, targets = [], []
        for idx in indices:
            class_indices = self._select_indices(
                y, low_range=idx, high_range=idx + 1
            )
     
            data_indices.append(class_indices)
            targets.append(y[class_indices])

        data_indices = (
            np.concatenate(data_indices).astype(np.int64)
            if len(data_indices) != 0
            else np.array([], dtype=np.int64)
        )
        targets = (
            np.concatenate(targets)
            if len(targets) != 0
            else np.array([], dtype=y.dtype)
        )
        appendent_data, appendent_targets = None, None
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            targets = np.concatenate((targets, appendent_targets))
            
        x_idxs = x_idxs[data_indices]
        dataset = self.Dataset_class(
            x_idxs,
            targets,
            trsf,
            self.use_path,
            self.aug if source == "train" and mode == "train" else 1,
            appendent_data=appendent_data,
        )
        if ret_data:
            data = dataset.get_all_data()
            return data, targets, dataset
        else:
            return dataset
        # for idx in indices:
        #     if m_rate is None:
        #         class_data, class_targets = self._select(
        #             x, y, low_range=idx, high_range=idx + 1
        #         )
        #     else:
        #         class_data, class_targets = self._select_rmm(
        #             x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
        #         )
        #     data.append(class_data)
        #     targets.append(class_targets)

        # if appendent is not None and len(appendent) != 0:
        #     appendent_data, appendent_targets = appendent
        #     data.append(appendent_data)
        #     targets.append(appendent_targets)
            
        # data, targets = np.concatenate(data), np.concatenate(targets)

        # if ret_data:
        #     return data, targets, DummyDataset(data, targets, trsf, self.use_path,self.aug if source == "train" and mode == "train" else 1)
        # else:
        #     return DummyDataset(data, targets, trsf, self.use_path,self.aug if source == "train" and mode == "train" else 1)


    def _setup_data(self, dataset_name, shuffle, seed):
        idata, self.Dataset_class = _get_metadata(dataset_name)
        idata.download_data()

        # Data
        # self._train_data, self._train_targets = idata.train_data, idata.train_targets
        # self._test_data, self._test_targets = idata.test_data, idata.test_targets
        # self._val_data, self._val_targets = idata.val_data, idata.val_targets
        self._train_data_idx = idata.train_data_idx
        self._test_data_idx = idata.test_data_idx
        self._val_data_idx = idata.val_data_idx
        self._all_targets = idata.targets
        
        self._train_targets = self._all_targets[self._train_data_idx]
        self._test_targets = self._all_targets[self._test_data_idx]
        self._val_targets = self._all_targets[self._val_data_idx]
        self.use_path = False

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        self._val_targets = _map_new_class_index(self._val_targets, self._class_order)
        
    
    
    
    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        if isinstance(x,np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    # def getlen(self, index):
    #     y = self._train_targets
    #     return np.sum(np.where(y == index))
    

    def _select_indices(self, y, low_range, high_range):
        return np.where(np.logical_and(y >= low_range, y < high_range))[0]
    
    def _materialize_dict_data(self, data, indices):
        return np.asarray(
            [{k: v[idx] for k, v in data.items()} for idx in indices],
            dtype=object,
        )


class DummyDataset(Dataset):
    # def __init__(self, data, labels, trsf, use_path=False, aug=1):
    def __init__(
        self,
        data_idxs,
        labels,
        trsf,
        use_path=False,
        aug=1,
        appendent_data=None,
    ):  
        # assert isinstance(data, dict), "Data type error!"
        self.aug = aug
        self.data_idxs = data_idxs
        self.appendent_data = appendent_data
        self.labels = labels
        self.trsf = trsf

        self._load_data()
        
    def __len__(self):
        return len(self.labels)

    
    def _load_data(self):
        pass
            


    def __getitem__(self, idx):
        pass

    
    # def __getitem__(self, idx):
    #     if self.aug == 1:
    #         if self.use_path:
    #             image = self.trsf(pil_loader(self.data[idx]))
    #         else:
    #             image = self.trsf(Image.fromarray(self.data[idx]))
    #         label = self.labels[idx]
    #         return idx, image, label
    #     else:
    #         if self.use_path:
    #             images = [self.trsf(pil_loader(self.data[idx])) for _ in range(self.aug)]
    #         else:
    #             images = [self.trsf(Image.fromarray(self.data[idx])) for _ in range(self.aug)]
    #         label = self.labels[idx]
    #         return idx, *images, label


class AVE_DummyDataset(DummyDataset):
    def __init__(
        self,
        data_idxs,
        labels,
        trsf,
        use_path=False,
        aug=1,
        appendent_data=None,
    ):
        super().__init__(data_idxs, labels, trsf, use_path, aug, appendent_data)
        self.m = 2

    def _load_data(self):
        self.video_features = np.load("data/AVE/video_features.npy", allow_pickle=True, mmap_mode="r")
        self.audio_features = np.load("data/AVE/audio_features.npy", allow_pickle=True, mmap_mode="r")
    
    def __getitem__(self, idx):
        base_len = len(self.data_idxs)
        if idx < base_len:
            real_idx = self.data_idxs[idx]
            sample = {"m1": self.video_features[real_idx], "m2": self.audio_features[real_idx]}
        else:
            mem_idx = idx - base_len
            sample = self.appendent_data[mem_idx]
        
        assert isinstance(sample, dict)
        sample = {k: torch.as_tensor(v) for k,v in sample.items()}
        return sample, self.labels[idx]
    
    def get_all_data(self):
        data = []
        for idx in tqdm(range(len(self))):
            sample, _ = self.__getitem__(idx)
            data.append(sample)
        return data


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_metadata(dataset_name):
    name = dataset_name.lower()
    if name == "ave":
        return AVE(), AVE_DummyDataset  
    elif name == "mmea-cl":
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


