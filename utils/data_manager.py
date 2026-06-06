import logging
import numpy as np
from numpy.lib.arraysetops import isin
from PIL import Image
from torch.nn.functional import instance_norm
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR10_AA, iCIFAR100_AA, AVE, MMEA_CL, Kinetics
from tqdm import tqdm
import os
from scipy import signal
import pandas as pd
from scipy.integrate import trapz
import torch
from numpy.random import randint
import h5py
from collections import OrderedDict

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
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        elif source == "val":
            x, y = self._val_data, self._val_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = self._train_trsf
        elif mode == "test" or mode == "val":
            trsf = self._test_trsf
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        # data, targets = [], []
        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
     
            data.append(class_data)
            targets.append(class_targets)

    
        targets = np.concatenate(targets) if len(targets) != 0 else np.array([])
        data = np.concatenate(data) if len(data) != 0 else np.array([])
        
        
        appendent_data, appendent_targets = None, None
        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            targets = np.concatenate((targets, appendent_targets))


        dataset = self.Dataset_class(
            data,
            targets,
            trsf,
            self.use_path,
            mode,
            appendent_data=appendent_data,
        )
        if ret_data:
            data = dataset.get_all_data()
            return data, targets, dataset
        else:
            return dataset
        

    def _setup_data(self, dataset_name, shuffle, seed):
        idata, self.Dataset_class = _get_metadata(dataset_name)
        idata.download_data()

        # Data
        # self._train_data, self._train_targets = idata.train_data, idata.train_targets
        # self._test_data, self._test_targets = idata.test_data, idata.test_targets
        # self._val_data, self._val_targets = idata.val_data, idata.val_targets
        self._train_data = idata.train_data
        self._test_data = idata.test_data
        self._val_data = idata.val_data


        self._train_targets = idata.train_targets
        self._test_targets = idata.test_targets
        self._val_targets = idata.val_targets
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
        data,
        labels,
        trsf,
        use_path=False,
        mode='train',
        appendent_data=None,
    ):  
        # assert isinstance(data, dict), "Data type error!"
        self.mode = mode
        self.data = data
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

    def get_all_data(self):
        data = []
        for idx in range(len(self)):
            sample, _ = self.__getitem__(idx)
            data.append(sample)
        return data



class AVE_DummyDataset(DummyDataset):
    def __init__(
        self,
        data_idxs,
        labels,
        trsf,
        use_path=False,
        mode='train',
        appendent_data=None,
    ):
        super().__init__(data_idxs, labels, trsf, use_path, mode, appendent_data)
        self.data_idxs = self.data
        self.m = 2

    def _load_data(self):
        self.video_features = np.load("data/AVE_features/video_features.npy", allow_pickle=True, mmap_mode="r")
        self.audio_features = np.load("data/AVE_features/audio_features.npy", allow_pickle=True, mmap_mode="r")
    
    def __getitem__(self, idx):
        base_len = len(self.data_idxs)
        if idx < base_len:
            real_idx = self.data_idxs[idx]
            sample = {"m1": self.video_features[real_idx], "m2": self.audio_features[real_idx]}
            sample = {k: torch.tensor(v) for k,v in sample.items()}
            # sample['m2'] = torch.mean(sample['m2'], dim=0)
        else:
            mem_idx = idx - base_len
            sample = self.appendent_data[mem_idx]
            # sample = {k: torch.tensor(v) for k,v in sample.items()}
        
        assert isinstance(sample, dict)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label



class Kinetics_DummyDataset(DummyDataset):
    def __init__(
        self,
        data,
        labels,
        trsf,
        use_path=False,
        mode='train',
        appendent_data=None,
    ):
        super().__init__(data, labels, trsf, use_path, mode, appendent_data)
        self.m = 2

    def _load_data(self):
        self.video_features = h5py.File("data/KS/visual_features.h5", 'r')
        self.audio_features = np.load("data/KS/audio_pretrained_feature_dict.npy", allow_pickle=True).item()
    
    def __getitem__(self, idx):
        base_len = len(self.data)
        if idx < base_len:
            real_id = self.data[idx]
            sample = {"m1": self.video_features[real_id][()], "m2": self.audio_features[real_id]}
            sample = {k: torch.from_numpy(v) for k,v in sample.items()}
            # sample['m2'] = torch.mean(sample['m2'], dim=0)
        else:
            mem_idx = idx - base_len
            sample = self.appendent_data[mem_idx]
            # sample = {k: torch.tensor(v) for k,v in sample.items()}
        
        assert isinstance(sample, dict)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label


class MMEA_DummyDataset(DummyDataset):
    def __init__(
        self,
        video_list,
        labels,
        trsf,
        use_path=False,
        mode='train',
        appendent_data=None,
    ):
        super().__init__(video_list, labels, trsf, use_path, mode, appendent_data)
        self.video_list = self.data
        self.m = 3
        self.transform = trsf
        self.num_segments = 8
        self.image_tmpl = {}
        
        self.modality = ["RGB", "Gyro", "Acce"]
        for m in self.modality:
            # Prepare dictionaries containing image name templates for each modality
            if m in ['RGB', 'RGBDiff']:
                self.image_tmpl[m] = "{:06d}.jpg"

        self.mpu_path = "/opt/data/private/PyCIL/data/UESTC-MMEA-CL/sensor/"

        self.new_length = OrderedDict({
                        ('RGB', 1),
                        ('Gyro', 24),
                        ('Acce', 24)
                    })
    
    def _mpu_data_convert(self, ori_mpu_data, acc_sensitivity=8192, gyro_sensitivity=16.4):

        acc_x = ori_mpu_data[0] / acc_sensitivity
        acc_y = ori_mpu_data[1] / acc_sensitivity
        acc_z = ori_mpu_data[2] / acc_sensitivity

        gyro_x = ori_mpu_data[3] / gyro_sensitivity
        gyro_y = ori_mpu_data[4] / gyro_sensitivity
        gyro_z = ori_mpu_data[5] / gyro_sensitivity

        return [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]

    def _median_filter(self, true_mpu_datas, kernel_size=3):
        filter_datas = []
        for i in range(6):
            filter_data = signal.medfilt(true_mpu_datas[:, i], kernel_size=kernel_size)
            filter_datas.append(filter_data)

        return np.array(filter_datas).astype(np.float32)

    def _trapz(self, filter_datas):
        mean = np.mean(filter_datas, axis=1).reshape((3, 1))
        filter_datas = filter_datas - mean
        angles = []
        init_angle = filter_datas[:, 0].reshape((1, 3))
        for i in range(1, filter_datas.shape[1]+1):
            angle = trapz(filter_datas[:, 0:i])
            angles.append(angle)
        angles = np.array(angles) + init_angle
        
        return angles.astype(np.float32)
    
    def _mpu_process(self, record):
        file_path = record.path.replace('/opt/data/private/PyCIL/data/UESTC-MMEA-CL/frame/', self.mpu_path) + '.csv'
        try:
            mpu_datas = pd.read_csv(file_path, header=None)
            true_mpu_datas = []
            for i in range(len(mpu_datas)):
                ori_data = mpu_datas.loc[i]
                true_mpu_data = self._mpu_data_convert(ori_data)
                true_mpu_datas.append(true_mpu_data)

            filter_datas = self._median_filter(np.array(true_mpu_datas), kernel_size=5)
            angles = self._trapz(filter_datas[3:6, :])

            self._process_acce_data = filter_datas[0:3, :].T
            self._process_gyro_data = angles
        except Exception:
            print('error loading gyro file:', file_path)    
    
    def my_load_data(self, modality, record, idx):
        if modality == 'RGB' or modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(record.path, self.image_tmpl[modality].format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(record.path, self.image_tmpl[modality].format(idx)))

        elif modality == 'Flow':
            try:
                x_img = Image.open(os.path.join(record.path, self.image_tmpl[modality].format('x', idx))).convert('L')
                y_img = Image.open(os.path.join(record.path, self.image_tmpl[modality].format('y', idx))).convert('L')
                return [x_img, y_img]
            except Exception:
                print('error loading flow image:', os.path.join(record.path, self.image_tmpl[modality].format('x/y', idx)))

        elif modality == 'Acce':
            file_path = record.path.replace('/opt/data/private/PyCIL/data/UESTC-MMEA-CL/frame/', self.mpu_path) + '.csv'
            try:
                mpu_data = self._process_acce_data[idx]
                return mpu_data
            except IndexError:
                print('error loading gyro file:', file_path, idx)

        elif modality == 'Gyro':
            file_path = record.path.replace('/opt/data/private/PyCIL/data/UESTC-MMEA-CL/frame/', self.mpu_path) + '.csv'
            try:
                mpu_data = self._process_gyro_data[idx]
                return mpu_data
            except IndexError:
                print('error loading gyro file:', file_path, idx)

    def _sample_indices(self, record, modality):
        """

        :param record: VideoRecord
        :return: list
        """
        average_duration = (record.num_frames[modality] - self.new_length[modality] + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        # elif record.num_frames[modality] > self.num_segments:
        #     offsets = np.sort(randint(record.num_frames[modality] - self.new_length[modality] + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets

    def _get_val_indices(self, record, modality):
        if record.num_frames[modality] > self.num_segments + self.new_length[modality] - 1:
            tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets

    def _get_test_indices(self, record, modality):

        tick = (record.num_frames[modality] - self.new_length[modality] + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):

        input = {}
        record = self.video_list[index]

        if 'Gyro' in self.modality or 'Acce' in self.modality:
            self._mpu_process(record)

        for m in self.modality:
            if self.mode == 'train':
                segment_indices = self._sample_indices(record, m)
            elif self.mode == 'val':
                segment_indices = self._get_val_indices(record, m)
            elif self.mode == 'test':
                segment_indices = self._get_test_indices(record, m)

            # We implement a Temporal Binding Window (TBW) with size same as the action's length by:
            #   1. Selecting different random indices (timestamps) for each modality within segments
            #      (this is similar to using a TBW with size same as the segment's size)
            #   2. Shuffling randomly the segments of Flow, Audio (RGB is the anchor hence not shuffled)
            #      which binds data across segments, hence making the TBW same in size as the action.
            #   Example of an action with 90 frames across all modalities:
            #    1. Synchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [12, 41, 80], Audio: [12, 41, 80]
            #    2. Asynchronous selection of indices per segment:
            #       RGB: [12, 41, 80], Flow: [9, 55, 88], Audio: [20, 33, 67]
            #    3. Asynchronous selection of indices per action:
            #       RGB: [12, 41, 80], Flow: [88, 55, 9], Audio: [67, 20, 33]

            if m != 'RGB' and self.mode == 'train':
                np.random.shuffle(segment_indices)

            img= self.get(m, record, segment_indices)
            input[m] = img

        return input, self.labels[index]

    def get(self, modality, record, indices):

        if modality != 'Gyro' and modality != 'Acce':
            images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length[modality]):
                    seg_imgs = self.my_load_data(modality, record, p+1)
                    images.extend(seg_imgs)
                    if p < record.num_frames[modality]:
                        p += 1
            process_data = self.transform[modality](images)
        
        elif modality == 'Acce':
            mpu_datas1 = []
            for seg_ind in indices:
                mpu_data1 = []
                p = int(seg_ind)
                for i in range(self.new_length[modality]):
                    single_mpu_data1 = self.my_load_data(modality, record, p)
                    mpu_data1.append(single_mpu_data1)
                    if p < record.num_frames[modality]:
                        p += 1
                
                mpu_datas1.append(np.array(mpu_data1))

            process_data = self.transform[modality](mpu_datas1)
        
        elif modality == 'Gyro':
            mpu_datas2 = []
            for seg_ind in indices:
                mpu_data2 = []
                p = int(seg_ind)
                for i in range(self.new_length[modality]):
                    single_mpu_data2 = self.my_load_data(modality, record, p)
                    mpu_data2.append(single_mpu_data2)
                    if p < record.num_frames[modality]:
                        p += 1

                mpu_datas2.append(np.array(mpu_data2))

            process_data = self.transform[modality](mpu_datas2)

        return process_data

    def __len__(self):
        return len(self.video_list)


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_metadata(dataset_name):
    name = dataset_name.lower()
    if name == "ave":
        return AVE(), AVE_DummyDataset  
    elif name == "mmea":
        return MMEA_CL(), MMEA_DummyDataset
    elif name == "kinetics":
        return Kinetics(), Kinetics_DummyDataset
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


