import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import My_Net
from collections import defaultdict
import os
from utils.toolkit import target2onehot, tensor2numpy
from utils import ddp
import wandb
import seaborn as sns
from itertools import cycle
import matplotlib.pyplot as plt
from tqdm.contrib import tzip



EPSILON = 1e-8


class AVCIL_My(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = My_Net(args, False)

    def after_task(self):
        # self._old_network = self._network.copy().freeze()
        self._old_network = torch.load(
            'save/{}/av_cil_task_{}_best_model.pkl'.format(self._dataset, self._cur_task),
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
            
    
    def _compute_accuracy(self, model, loader, old_model=None):
        model.eval()


        correct, total = 0, 0
        test_losses = 0.0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts == targets).sum()
            total += len(targets)
            
            # test_losses += loss.item()

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    
    def get_loss(self, data, labels, exemplar_data, exemplar_labels):
        details = {}
        data_batch_size = labels.shape[0]
        exemplar_data_batch_size = exemplar_labels.shape[0]

        visual = data["m1"]
        audio = data["m2"]
        exemplar_visual = exemplar_data["m1"]
        exemplar_audio = exemplar_data["m2"]

        total_visual = torch.cat((visual, exemplar_visual))
        total_audio = torch.cat((audio, exemplar_audio))
        total_visual = total_visual.to(self._device)
        total_audio = total_audio.to(self._device)

        inputs = {"m1": total_visual,
                  "m2": total_audio}
        outputs = self._network(inputs, out_feature_before_fusion=True, out_attn_score=True)
        out = outputs["logits"]
        audio_feature = outputs["audio_feature"]
        visual_feature = outputs["visual_feature"]
        spatial_attn_score = outputs["spatial_attn_score"]
        temporal_attn_score = outputs["temporal_attn_score"]
        
        # out, audio_feature, visual_feature, spatial_attn_score, temporal_attn_score = self._network(visual=total_visual, audio=total_audio, out_feature_before_fusion=True, out_attn_score=True)
        with torch.no_grad():
            old_outputs = self._old_network(inputs, out_attn_score=True)
            old_out, old_spatial_attn_score, old_temporal_attn_score = old_outputs["logits"], old_outputs["spatial_attn_score"], old_outputs["temporal_attn_score"]
            old_spatial_attn_score = old_spatial_attn_score.detach()
            old_temporal_attn_score = old_temporal_attn_score.detach()

        # if args.instance_contrastive:
        instance_contra_loss = self.cal_contrastive_loss(
            audio_feature,
            visual_feature,
            temperature=self.args["instance_contrastive_temperature"],
        )
                
        # if args.class_contrastive:
        all_labels = torch.cat((labels, exemplar_labels))
        class_contra_loss = self.class_contrastive_loss(
            audio_feature,
            visual_feature,
            all_labels,
            temperature=self.args["class_contrastive_temperature"],
        )
        
        # if args.attn_score_distil:
        exem_spatial_attn_score = spatial_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(2, 3)
        exem_spatial_attn_score = exem_spatial_attn_score.reshape(-1, exem_spatial_attn_score.shape[-1])

        exem_old_spatial_attn_score = old_spatial_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(2, 3)
        exem_old_spatial_attn_score = exem_old_spatial_attn_score.reshape(-1, exem_old_spatial_attn_score.shape[-1])

        exem_temporal_attn_score = temporal_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(1, 2)
        exem_temporal_attn_score = exem_temporal_attn_score.reshape(-1, exem_temporal_attn_score.shape[-1])

        exem_old_temporal_attn_score = old_temporal_attn_score[data_batch_size:data_batch_size+exemplar_data_batch_size].transpose(1, 2)
        exem_old_temporal_attn_score = exem_old_temporal_attn_score.reshape(-1, exem_old_temporal_attn_score.shape[-1])

        spatial_attn_dist_loss = F.kl_div(exem_spatial_attn_score.log(), exem_old_spatial_attn_score, reduction='sum') / exemplar_data_batch_size
        temporal_attn_dist_loss = F.kl_div(exem_temporal_attn_score.log(), exem_old_temporal_attn_score, reduction='sum') / exemplar_data_batch_size


        last_step_out_class_num = self._known_classes
        class_num_per_step = self._increment
        old_out = old_out[:,:last_step_out_class_num]
        
        curr_out = out[:data_batch_size, last_step_out_class_num:]
        curr_labels = labels - last_step_out_class_num
        loss_curr = self.CE_loss(class_num_per_step, curr_out, curr_labels)

        prev_out = out[data_batch_size:data_batch_size+exemplar_data_batch_size, :last_step_out_class_num]
        loss_prev = self.CE_loss(last_step_out_class_num, prev_out, exemplar_labels)

        loss_CE = (loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size) / (data_batch_size + exemplar_data_batch_size)

        # if self._dataset == 'AVE' and args.class_num_per_step == 4 and step == 1:
        #     loss_CE = CE_loss(args.class_num_per_step + last_step_out_class_num, out, torch.cat((labels, exemplar_labels)))

        loss_KD = torch.zeros(self._cur_task).to(self._device)
        
        for t in range(self._cur_task):
            start = t * class_num_per_step
            end = (t + 1) * class_num_per_step

            soft_target = F.softmax(old_out[:, start:end] / self.args["T"], dim=1)
            output_log = F.log_softmax(out[:, start:end] / self.args["T"], dim=1)
            loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (self.args["T"]**2)
        loss_KD = loss_KD.sum()
        loss = loss_CE + loss_KD
        # if args.instance_contrastive:
        loss += 0.5 * instance_contra_loss
        # if args.class_contrastive:
        loss += 1.0* class_contra_loss
        # if args.attn_score_distil:
        loss += 0.5 * spatial_attn_dist_loss + (1 - 0.5) * temporal_attn_dist_loss

        return loss
    
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
            batch_size=self.args["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=self.args["num_workers"],
        )
        if self._cur_task > 0:
            mem_set = data_manager.get_dataset(
                [],
                source='train',
                mode='train',
                appendent=self._get_memory()
            )
            mem_sampler = ddp.make_sampler(mem_set, shuffle=True)
            self.mem_loader = DataLoader(
                mem_set,
                batch_size=self.args["batch_size"],
                shuffle=mem_sampler is None,
                sampler=mem_sampler,
                num_workers=self.args["num_workers"],
            )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"]
        )
        
        val_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="val", mode="val"
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"]
        )
        
        if self._cur_task > 0:
            mem_dataset = data_manager.get_dataset(
                [],
                source='train',
                mode='train',
                appendent=self._get_memory()
            )
            
            mem_sampler = ddp.make_sampler(mem_dataset, shuffle=False)
            self.mem_loader = DataLoader(
                mem_dataset,
                batch_size=self.args["batch_size"],
                shuffle=False,
                sampler=mem_sampler,
                num_workers=self.args["num_workers"],
            )

        self._network = ddp.wrap_model(self._network, self._device, self.args)
        self._train(self.train_loader, self.val_loader)
        self._network = ddp.unwrap_model(self._network)
        ddp.barrier()
        self._network = torch.load(
            'save/{}/av_cil_task_{}_best_model.pkl'.format(self._dataset, self._cur_task),
            map_location=self._device,
        )
        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        # if self._cur_task > 0:
            # self._network.weight_align(self._total_classes - self._known_classes)
    
        
    def cal_contrastive_loss(self, feature_1, feature_2, temperature=0.1):
        # (BS, BS)
        score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
        num_sample = score.shape[0]
        label = torch.arange(num_sample).to(score.device)

        loss = self.CE_loss(num_sample, score, label)
        return loss
    
    def CE_loss(self, num_classes, logits, label):
        if torch.any(label < 0) or torch.any(label >= num_classes):
            raise ValueError(
                "label out of range for CE_loss: "
                f"num_classes={num_classes}, "
                f"label_min={label.min().item()}, label_max={label.max().item()}"
            )
        targets = F.one_hot(label, num_classes=num_classes)
        loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * targets, dim=1))

        return loss
    
    def class_contrastive_loss(self, feature_1, feature_2, label, temperature=0.1):
        class_matrix = label.unsqueeze(0)
        class_matrix = class_matrix.repeat(class_matrix.shape[1], 1)
        class_matrix = class_matrix == label.unsqueeze(-1)
        # (BS, BS)
        class_matrix = class_matrix.float()
        # (BS, BS)
        score = torch.mm(feature_1, feature_2.transpose(0, 1)) / temperature
        loss = -torch.mean(torch.mean(F.log_softmax(score, dim=-1) * class_matrix, dim=-1))
        ###################################################################################################
        # You can also use the following implementation, which is more consistent with Equation (7) in our paper, 
        # but you may need to further adjust the hyperparameters lam_I and lam_C to get optimal performance.
        # loss = -torch.mean(
        #     (torch.sum(F.log_softmax(score, dim=-1) * class_matrix, dim=-1)) / torch.sum(class_matrix, dim=-1))
        ###################################################################################################

        return loss


    def _train(self, train_loader, val_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            )
            self._init_train(train_loader, val_loader, optimizer, scheduler)
        else:
            optimizer = optim.Adam(
                self._network.parameters(),
                lr=self.args["lrate"],
                weight_decay=self.args["weight_decay"],
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            )
            self._update_representation(train_loader, val_loader, optimizer, scheduler)

    def _init_train(self, train_loader, val_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]), disable=not ddp.is_main_process())
        best_acc = -1e9
        for _, epoch in enumerate(prog_bar):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (inputs, targets) in enumerate(train_loader):
                inputs, targets = {k:v.to(self._device) for k,v in inputs.items()}, targets.to(self._device)
                # logits = self._network(inputs)["logits"]
                logits = self._network(inputs)["logits"]
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
                val_acc = self._compute_accuracy(self._network, val_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    val_acc,
                )
                if val_acc > best_acc:
                    save_dir = os.path.join("save", self._dataset)
                    if ddp.is_main_process():
                        os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, 'av_cil_task_{}_best_model.pkl'.format(self._cur_task))
                    if ddp.is_main_process():
                        torch.save(ddp.unwrap_model(self._network), save_path)
                    best_acc = val_acc
                    if ddp.is_main_process():
                        print(f"save best model at epoch {epoch}")
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, val_loader, optimizer, scheduler):
        # prog_bar = tqdm(range(epochs))
        prog_bar = tqdm(range(self.args["epochs"]), disable=not ddp.is_main_process())
        best_acc = -1e9
        for _, epoch in enumerate(prog_bar):
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            if hasattr(self.mem_loader.sampler, "set_epoch"):
                self.mem_loader.sampler.set_epoch(epoch)
            self._network.train()
            loss_details = defaultdict(float)
            correct, total = 0, 0
            iterator = tzip(train_loader, cycle(self.mem_loader))
            for samples in iterator:
                curr, prev = samples
                data, labels = curr
                labels = labels.to(self._device)
                exemplar_data, exemplar_labels = prev
                exemplar_labels = exemplar_labels.to(self._device)
                loss = self.get_loss(data, labels, exemplar_data, exemplar_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_details['tot_loss'] += loss.item()
            
            adjust_learning_rate(optimizer, epoch, self.args["milestones"])
            # train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            val_acc = self._compute_accuracy(self._network, val_loader)
            
            if val_acc > best_acc:
                save_dir = os.path.join("save", self._dataset)
                if ddp.is_main_process():
                    os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, 'av_cil_task_{}_best_model.pkl'.format(self._cur_task))
                if ddp.is_main_process():
                    torch.save(ddp.unwrap_model(self._network), save_path)
                best_acc = val_acc
                if ddp.is_main_process():
                    print(f"save best model at epoch {epoch} with acc {val_acc}")

            if ddp.is_main_process():
                wandb.log({
                    # f"train/task_{self._cur_task}_acc": train_acc,
                    f"train/task_{self._cur_task}_loss" : loss_details['tot_loss'] / len(train_loader),
                    # f"train/task_{self._cur_task}_CE_loss" : loss_details['CE_loss'] / len(train_loader),
                    # f"train/task_{self._cur_task}_KD_loss" : loss_details['KD_loss'] / len(train_loader),
                    # f"train/task_{self._cur_task}_Router_KD_loss" : loss_details['Router_KD_loss'] / len(train_loader),
                    f"eval/task_{self._cur_task}_acc" : val_acc,
                    # f"eval/task_{self._cur_task}_loss" : val_loss / len(val_loader)
                })
            
            # prog_bar.set_description(info)
        # self.visualize_logits(self._network, self.test_loader)
        # logging.info(info)
        


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def adjust_learning_rate(optimizer, epoch, milestones):
    miles_list = np.array(milestones) - 1
    if epoch in miles_list:
        current_lr = optimizer.param_groups[0]['lr']
        new_lr = current_lr * 0.1
        print('Reduce lr from {} to {}'.format(current_lr, new_lr))
        for param_group in optimizer.param_groups: 
            param_group['lr'] = new_lr
