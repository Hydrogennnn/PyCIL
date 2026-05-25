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




EPSILON = 1e-8

init_epoch = 100
init_lr = 1e-3
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 100
lrate = 1e-3
milestones = [80, 120]
lrate_decay = 0.1
batch_size = 128
weight_decay = 1e-4
num_workers = 0
T = 2


class MoE(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MoENet(args, False)
        # Weights for the DirMoE continual-learning regularizers.
        self.route_prior_weight = args.get("route_prior_weight", 0.1)
        self.route_sparsity_weight = args.get("route_sparsity_weight", 0.01)

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._old_network = torch.load(
            'save/{}/task_{}_best_model.pkl'.format(self._dataset, self._cur_task),
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
    
    
    
    def get_loss(self, inputs, targets):
        details = {
            "KD_loss": 0.0,
            "Router_KD_loss": 0.0,
            "Dirichlet_prior_loss": 0.0,
            "Dirichlet_KL_loss": 0.0,
            "Bernoulli_KL_loss": 0.0,
            "Route_sparsity_loss": 0.0,
        }
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
        # DirMoE routing prior: constrain old-class exemplars with the previous
        # task posterior distribution, and keep the expected active expert count sparse.
        loss_route_prior = torch.zeros((), device=self._device)
        loss_route_sparsity = torch.zeros((), device=self._device)
        idxes = torch.where(targets < self._known_classes)[0]
        net = ddp.unwrap_model(self._network)
        if self.route_sparsity_weight > 0:
            # Apply to all samples so the relaxed router remains close to top-k.
            loss_route_sparsity = net.sparsity_loss(inputs)
            details["Route_sparsity_loss"] = loss_route_sparsity.item()

        if self._old_network is not None and len(idxes) != 0:
            # Only old-class rehearsal samples have a previous-task posterior;
            # use it as the prior to reduce routing drift and forgetting.
            old_inputs = {k:v[idxes] for k,v in inputs.items()}
            with torch.no_grad():
                old_alpha, old_select_probs = self._old_network.get_route_params(old_inputs)
                old_alpha = old_alpha.detach()
                old_select_probs = old_select_probs.detach()
            loss_route_prior, route_details = net.dirichlet_prior_loss(
                old_inputs,
                prior_alpha=old_alpha,
                prior_select_probs=old_select_probs,
            )
            details["Dirichlet_prior_loss"] = loss_route_prior.item()
            details["Dirichlet_KL_loss"] = route_details["dirichlet_kl"].item()
            details["Bernoulli_KL_loss"] = route_details["bernoulli_kl"].item()
            details["Router_KD_loss"] = loss_route_prior.item()

        loss = (
            loss_clf
            + 0.5 * loss_KD
            + self.route_prior_weight * loss_route_prior
            + self.route_sparsity_weight * loss_route_sparsity
        )
        
        
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
            appendent=self._get_memory(),
        )
        train_sampler = ddp.make_sampler(train_dataset, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        val_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="val", mode="val"
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
        self._network = torch.load(
            'save/{}/task_{}_best_model.pkl'.format(self._dataset, self._cur_task),
            map_location=self._device,
        )
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if self._cur_task > 0:
            self._network.weight_align(self._total_classes - self._known_classes)
    
        


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
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
                loss, details = self.get_loss(inputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                logits = details["logits"]
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
                    f"train/task_{self._cur_task}_Dirichlet_prior_loss" : loss_details['Dirichlet_prior_loss'] / len(train_loader),
                    f"train/task_{self._cur_task}_Dirichlet_KL_loss" : loss_details['Dirichlet_KL_loss'] / len(train_loader),
                    f"train/task_{self._cur_task}_Bernoulli_KL_loss" : loss_details['Bernoulli_KL_loss'] / len(train_loader),
                    f"train/task_{self._cur_task}_Route_sparsity_loss" : loss_details['Route_sparsity_loss'] / len(train_loader),
                    f"eval/task_{self._cur_task}_acc" : val_acc,
                    f"eval/task_{self._cur_task}_loss" : val_loss / len(val_loader)
                })
            
            prog_bar.set_description(info)
        if ddp.is_main_process():
            self.visualize_logits(self._network, self.test_loader)
            self.visualize_gating(self._network, self.test_loader)
        logging.info(info)
        


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
