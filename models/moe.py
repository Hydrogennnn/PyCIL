import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import MoENet
from collections import defaultdict
import os
from utils.toolkit import target2onehot, tensor2numpy
from utils import ddp
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import sys
import math

EPSILON = 1e-8

init_epoch = 100
init_lr = 1e-3
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 50
lrate = 1e-3
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 32
weight_decay = 1e-4
num_workers = 4
T = 2

ca_epochs = 20
ca_lr = 0.001
ca_batchsize = 24


class MoE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MoENet(args, False)

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._old_network = torch.load(
            'save/{}/task_{}_final_model.pkl'.format(self._dataset, self._cur_task),
            map_location=self._device,
        )
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        
    def visualize_logits(self, model, loader):
        model.eval()
        y_pred = []
        y_true = []
        res = []
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=1, dim=1, largest=True, sorted=True
            )[1]
            res.append(outputs.cpu().numpy())
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        
        res = np.concatenate(res)
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        
        assert len(y_pred) == len(y_true), "Data length error."
        
        increment = self._increment
        for class_id in range(0, np.max(y_true), increment):
            idxes = np.where(
                np.logical_and(y_true >= class_id, y_true < class_id + increment)
            )[0]
            
            label = "{}-{}".format(
                str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
            )
            print("=========", "Class Intervals:",label,"=========")
            preds_in_task = y_pred[idxes].flatten()
            unique, counts = np.unique(preds_in_task, return_counts=True)
            for cls, cnt in sorted(zip(unique, counts)):
                print(f"  class {cls:>3d}: {cnt} 个样本")
            # with np.printoptions(threshold=np.inf):
            #     print(res[idxes])
            
            # all_acc[label] = np.around(
            #     (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            # )
            
    def visualize_gating(self, model, loader):
        model.eval()
        all_loads = []
        y_true = []
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
            with torch.no_grad():
                load = ddp.unwrap_model(model).get_load(inputs)
            all_loads.append(load.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        
        all_loads = np.concatenate(all_loads)
        y_true = np.concatenate(y_true)
        increment = self._increment
        
        
        loads_per_task = []
        for class_id in range(0, self._total_classes, increment):
            idxes = np.where(
                np.logical_and(y_true >= class_id, y_true < class_id + increment)
            )[0]
            load_cur_task = all_loads[idxes].sum(0)
            loads_per_task.append(load_cur_task)
        
        loads_per_task = np.stack(loads_per_task)
        # print(loads_per_task.shape)
        
        
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(loads_per_task, cmap="viridis")
        plt.title("Token-Expert Routing Heatmap")
        plt.xlabel("Expert")
        plt.ylabel("Token")
        plt.savefig(os.path.join(f"save/{self._dataset}",f'load_{self._cur_task}.png'))
                
            # with np.printoptions(threshold=np.inf):
            #     print(res[idxes])
            
            # all_acc[label] = np.around(
            #     (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            # )
        
    def _compute_accuracy(self, model, loader, old_model=None):
        model.eval()
        correct, total = 0, 0
        test_losses = 0.0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
            with torch.no_grad():
                loss, details = self.get_loss(inputs, targets)
                # for k,v in details.items():
                #     if 'loss' in k:
                #         print(k, v)
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == targets).sum()
            total += len(targets)
            
            test_losses += loss.item()

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2), test_losses / len(loader)
    
    def train_task_adaptive_prediction(self, model):
        prog_bar = tqdm(range(ca_epochs), disable=not ddp.is_main_process())
        model.train()
        crct_num = self._total_classes
        fc_params = ddp.unwrap_model(model).fc.parameters()
        ca_optimizer = optim.AdamW(
                fc_params,
                lr=ca_lr,
                weight_decay=weight_decay
                )
        
        ca_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=ca_optimizer, T_max=ca_epochs)

        for epoch in prog_bar:
            sampled_data = []
            sampled_label = []
            
            num_sampled_pcls = ca_batchsize * 5

            for c_id in range(self._total_classes):
                for cluster in range(len(self.cls_mean[c_id])):
                    mean = self.cls_mean[c_id][cluster]
                    var = self.cls_cov[c_id][cluster]

                    # 空簇或退化簇可能产生 0 方差。
                    # 这种高斯分布要么无效，要么基本不提供有效信息，所以跳过。
                    if var.mean() == 0:
                        continue

                    # 用该簇的方差构造对角协方差矩阵。
                    # 额外加上 1e-4 * I，保证协方差矩阵足够正定，
                    # 避免 MultivariateNormal 因数值问题报错。
                    m = MultivariateNormal(mean.float(), (torch.diag(var) + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)).float())
                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)
                    sampled_label.extend([c_id] * num_sampled_pcls)
            
            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            
            inputs = sampled_data
            targets = sampled_label
            # 打乱合成特征，让每个 mini-batch 尽量混合不同类别和不同 centroid。
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            tot_loss = 0.0
            for _iter in range(crct_num):
                # 从合成特征池中切出一个 mini-batch。
                # 注意：这里复用了 num_sampled_pcls 作为 mini-batch size。
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                # fc_only=True 表示 inp 已经是提取好的 pre_logits 特征。
                # 模型会跳过 ViT/prompt 的前向过程，只执行分类器侧的层来产生 logits。
                outputs = model(inp, fc_only=True)
                logits = outputs['logits']

                # 对合成特征做标准交叉熵训练。
                # 这就是分类器校准的核心目标：重新平衡所有已见类别的 logits。
                loss = F.cross_entropy(logits, tgt)  # base criterion (CrossEntropyLoss)

                if not math.isfinite(loss.item()):
                    print("Loss is {}, stopping training".format(loss.item()))
                    sys.exit(1)

                ca_optimizer.zero_grad()
                loss.backward()
                ca_optimizer.step()
                tot_loss += loss.item()
                torch.cuda.synchronize()
            
            ca_scheduler.step()

            info = "Adapt FC, Epoch {}/{} => Loss {:.3f}".format(
                    epoch + 1,
                    ca_epochs,
                    tot_loss / crct_num
                )

            prog_bar.set_description(info)

    def get_loss(self, inputs, targets):
        details = {}
        logits = self._network(inputs)["logits"]
        loss_clf = F.cross_entropy(logits, targets)
        
        # logit_KD
        loss_KD = 0.0
        if self._old_network is not None:
            with torch.no_grad():
                old_logits = self._old_network(inputs)["logits"]
                old_logits = old_logits.detach()
            # logit KD
            loss_KD = torch.zeros(self._cur_task).to(self._device)
            for t in range(self._cur_task):
                start = t * self._increment
                end = (t + 1) * self._increment

                output_log = F.log_softmax(logits[:, start:end] / T, dim=1)
                soft_target = F.softmax(old_logits[:, start:end] / T, dim=1)
                loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (T**2)
            loss_KD = loss_KD.sum()
            details["KD_loss"] = loss_KD.item()
        # Router_KD
        
        loss_route_kd = 0.0
        idxes = torch.where(targets < self._known_classes)[0]
        if self._old_network is not None and len(idxes)!= 0:
            inputs = {k:v[idxes] for k,v in inputs.items()}
            route_score= ddp.unwrap_model(self._network).get_gating(inputs)
            with torch.no_grad():
                old_route_score = self._old_network.get_gating(inputs)
                
            log_route_score = F.log_softmax(route_score, dim=-1)
            old_route_score = F.softmax(old_route_score, dim=-1)

            loss_route_kd = F.kl_div(log_route_score, old_route_score, reduction='batchmean')
            
            details["Router_KD_loss"] = loss_route_kd.item()
            # if not self._network.training:
            #     print(len(idxes))
            #     print(targets)
                # print("route_score nan:", route_score.isnan().any().item())
                # print("old_route_score nan:", old_route_score.isnan().any().item())
                # print("route_score range:", route_score.min().item(), route_score.max().item())
                # print("idxes len:", len(idxes))
        loss = loss_clf + 0.5*loss_KD + loss_route_kd
        
        
        details["CE_loss"] = loss_clf.item()
        details["tot_loss"] = loss.item()
        details["logits"] = logits
        
        return loss, details
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes - 1)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            # appendent=self._get_memory(),
        )
        train_sampler = ddp.make_sampler(train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        val_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="val", mode="val"
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        if self._cur_task > 0:
            mem_dataset = data_manager.get_dataset(
                [],
                source='train',
                mode='train',
                appendent=self._get_memory()
            )
            
            self.mem_loader = DataLoader(
                mem_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            
            
        

        self._network = ddp.wrap_model(self._network, self._device, self.args)
        self._train(self.train_loader, self.val_loader)
        self._network = ddp.unwrap_model(self._network)
        ddp.barrier()
        # self._network = torch.load(
        #     'save/{}/task_{}_best_model.pkl'.format(self._dataset, self._cur_task),
        #     map_location=self._device,
        # )
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self.build_multi_centroid(data_manager)

        if ddp.is_main_process():
            self.visualize_logits(self._network, self.test_loader)
        if self._cur_task > 0:
            # self._network.weight_align(self._total_classes - self._known_classes)
            self._network = ddp.wrap_model(self._network, self._device, self.args)
            self.train_task_adaptive_prediction(self._network)
            self._network = ddp.unwrap_model(self._network)

        if ddp.is_main_process():
            save_dir = os.path.join("save", self._dataset)
            save_path = os.path.join(save_dir, 'task_{}_final_model.pkl'.format(self._cur_task))
            torch.save(ddp.unwrap_model(self._network), save_path)
            self.visualize_logits(self._network, self.test_loader)
        


    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, val_loader, optimizer, scheduler)
        else:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=lrate,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, val_loader, optimizer, scheduler)

    def _init_train(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch), disable=not ddp.is_main_process())
        best_acc = -1e9
        for _, epoch in enumerate(prog_bar):

            if self._dataset == "mmea":
                self._network.feature_extractor.freeze_fn('partialbn_statistics')
                self._network.feature_extractor.freeze_fn('bn_statistics')

            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                print("forwarding batch {}, epoch {}".format(i, epoch))
                inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
                
                logits = self._network(inputs)["logits"]
                print("forwarded batch {}, epoch {}".format(i, epoch))
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                val_acc, _ = self._compute_accuracy(self._network, val_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    val_acc,
                )
                if val_acc > best_acc:
                    save_dir = os.path.join("save", self._dataset)
                    if ddp.is_main_process():
                        os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'task_{}_best_model.pkl'.format(self._cur_task))
                    if ddp.is_main_process():
                        torch.save(ddp.unwrap_model(self._network), save_path)
                    best_acc = val_acc
                    if ddp.is_main_process():
                        print(f"save best model at epoch {epoch}")
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs), disable=not ddp.is_main_process())
        best_acc = -1e9
        for _, epoch in enumerate(prog_bar):
            if self._dataset == "mmea":
                self._network.feature_extractor.freeze_fn('partialbn_statistics')
                self._network.feature_extractor.freeze_fn('bn_statistics')

            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            self._network.train()
            loss_details = defaultdict(float)
            correct, total = 0, 0
            
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
                loss, details = self.get_loss(inputs, targets)
                # logits = self._network(inputs)["logits"]

                # loss_clf = F.cross_entropy(logits, targets)
                # # loss_kd = _KD_loss(
                # #     logits[:, : self._known_classes],
                # #     self._old_network(inputs)["logits"],
                # #     T,
                # # )

                # # loss = loss_clf + loss_kd
                # loss = loss_clf
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # losses += loss.item()
                for k, v in details.items():
                    if 'loss' not in k:
                        continue
                    loss_details[k] += v
                logits = details['logits']
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            val_acc, val_loss = self._compute_accuracy(self._network, val_loader)
            
            if val_acc > best_acc:
                save_dir = os.path.join("save", self._dataset)
                if ddp.is_main_process():
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'task_{}_best_model.pkl'.format(self._cur_task))
                if ddp.is_main_process():
                    torch.save(ddp.unwrap_model(self._network), save_path)
                best_acc = val_acc
                if ddp.is_main_process():
                    print(f"save best model at epoch {epoch} with acc {val_acc}")
            
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Val_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                loss_details['tot_loss'] / len(train_loader),
                train_acc,
                val_acc,
            )
            if ddp.is_main_process():
                wandb.log({
                    f"train/task_{self._cur_task}_acc": train_acc,
                    f"train/task_{self._cur_task}_loss" : loss_details['tot_loss'] / len(train_loader),
                    f"train/task_{self._cur_task}_CE_loss" : loss_details['CE_loss'] / len(train_loader),
                    f"train/task_{self._cur_task}_KD_loss" : loss_details['KD_loss'] / len(train_loader),
                    f"train/task_{self._cur_task}_Router_KD_loss" : loss_details['Router_KD_loss'] / len(train_loader),
                    f"eval/task_{self._cur_task}_acc" : val_acc,
                    f"eval/task_{self._cur_task}_loss" : val_loss / len(val_loader)
                })
            
            prog_bar.set_description(info)
        
        logging.info(info)
        


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
