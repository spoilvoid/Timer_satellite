import os
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from opacus import PrivacyEngine

from data_provider.data_factory import data_provider, concat_data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.freeze_layer:
            model = self._freeze_model(model)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        return model
    
    def _freeze_model(self, model):
        if self.args.model == 'Timer_multivariate':
            for param in model.backbone.proj.parameters():
                param.requires_grad = False
            for param in model.backbone.patch_embedding.parameters():
                param.requires_grad = False
        else:
            print("Freezing is not supported for this model type.")
        return model
        

    def _get_data(self, flag):
        if self.args.model == 'Timer_multivariate' and self.args.data == 'multivariate':
            if flag == 'test':
                if hasattr(self, 'test_dataset') and hasattr(self, 'test_dataloader'):
                    pass
                else:
                    test_dataset, test_dataloader = concat_data_provider(self.args, flag)
                    self.test_dataset = test_dataset
                    self.test_dataloader = test_dataloader
                return self.test_dataset, self.test_dataloader
            else:
                if hasattr(self, 'train_dataset') and hasattr(self, 'train_dataloader') and hasattr(self, 'val_dataset') and hasattr(self, 'val_dataloader'):
                    pass
                else:
                    train_dataset, train_dataloader, val_dataset, val_dataloader = concat_data_provider(self.args, flag)
                    self.train_dataset = train_dataset
                    self.train_dataloader = train_dataloader
                    self.val_dataset = val_dataset
                    self.val_dataloader = val_dataloader
                if flag == 'train':
                    return self.train_dataset, self.train_dataloader
                elif flag == 'val':
                    return self.val_dataset, self.val_dataloader
                else:
                    raise ValueError('invalid set type')        
        else:
            data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :-self.args.pred_len, :], dec_inp], dim=1).float()

                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)

                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_y = batch_y[:, :, :self.args.n_pred_vars]

                if self.args.use_ims and flag == 'vali':
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss = criterion(outputs, batch_y)
                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def finetune(self, setting):
        finetune_data, finetune_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if self.args.train_test:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)

        if self.args.use_opacus:
            privacy_engine = PrivacyEngine()
            self.model, model_optim, finetune_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=model_optim,
                data_loader=finetune_loader,
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
        )

        if self.args.record_info:
            grad_norms = []
            if self.args.use_opacus:
                epsilons = []
        for epoch in range(self.args.finetune_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :-self.args.pred_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)

                # print("outputs: ", outputs.shape)
                # print("batch_y: ", batch_y.shape)
                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_y = batch_y[:, :, :self.args.n_pred_vars]

                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    # only use the forecast window to calculate loss
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss_val += loss
                count += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                if self.args.record_info:
                    total_norm = torch.norm(torch.stack([p.grad.norm() for p in self.model.parameters() if p.grad is not None])).item()
                    grad_norms.append(total_norm)
                    if self.args.use_opacus:
                        eps = privacy_engine.get_epsilon(delta=1e-5)
                        epsilons.append(eps)
                model_optim.step()
                torch.cuda.empty_cache()
            
            if self.args.use_opacus:
                eps = privacy_engine.get_epsilon(delta=1e-5)
                print(f"Epoch {epoch+1}: cost time: {time.time() - epoch_time:.2f}s, loss={loss.item():.4f}, ε={eps:.2f}, delta={1e-5}")
            else:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)

        if self.args.record_info:
            np.save(os.path.join(path, "grad_norms.npy"), grad_norms)
            if self.args.use_opacus:
                np.save(os.path.join(path, "epsilons.npy"), epsilons)

        best_model_path = os.path.join(path, "checkpoint.pth")
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        folder_path = os.path.join(self.args.test_dir, setting)
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        if self.args.use_ims:
            predict_length = self.args.seq_len
        else:
            predict_length = self.args.pred_len
        test_data, test_loader = self._get_data(flag='test')
        random_idx = np.random.randint(0, int(len(test_data)/ dist.get_world_size()))
        criterion = self._select_criterion()
        criterion_step = nn.MSELoss(reduction='none')
        total_loss_step = []
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :-self.args.pred_len, :], dec_inp], dim=1).float()

                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)

                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_y = batch_y[:, :, :self.args.n_pred_vars]

                if self.args.use_ims:
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                    loss_step = criterion_step(outputs[:, -self.args.seq_len:, :], batch_y).mean(dim=(0,2))
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])
                    loss_step = criterion_step(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :]).mean(dim=(0,2))

                loss_step = loss_step.detach().cpu()
                loss = loss.detach().cpu()
                total_loss_step.append(loss_step)
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])

                if i*test_loader.batch_size <= random_idx < (i+1)*test_loader.batch_size:
                    batch_ptr = random_idx % test_loader.batch_size
                    for feat_idx in range(outputs.shape[-1]):
                        if self.args.use_ims:
                            pred_feat = outputs[batch_ptr, -self.args.seq_len:, feat_idx].cpu().numpy()
                            true_feat = batch_y[batch_ptr, -self.args.seq_len:, feat_idx].cpu().numpy()
                        else:
                            pred_feat = outputs[batch_ptr, -self.args.pred_len:, feat_idx].cpu().numpy()
                            true_feat = batch_y[batch_ptr, -self.args.pred_len:, feat_idx].cpu().numpy()
                        
                        plt.scatter(range(1,predict_length+1), pred_feat, s=10, c= 'red',label="Pred")
                        plt.plot(range(1,predict_length+1), true_feat, label="GT")
                        plt.title(f"Prediction and GroundTurth of Feature {feat_idx + 1}")
                        plt.xlabel("Prediction Step")
                        plt.ylabel("Elec value")
                        plt.legend()
                        plt.savefig(os.path.join(folder_path, f"feat_{feat_idx+1}_of_test_data_{random_idx}th.png"))
                        plt.clf()
                
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss_step = torch.tensor(np.average(total_loss_step, weights=total_count, axis=0))
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss_step = total_loss_step / dist.get_world_size()
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss_step = np.average(total_loss_step, weights=total_count, axis=0)
            total_loss = np.average(total_loss, weights=total_count)

        plt.scatter(range(1,predict_length+1), total_loss_step, s=10)
        plt.title(f"loss in test set")
        plt.xlabel("Prediction Step")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(os.path.join(folder_path, f"test_data_mse.png"))
        plt.clf()     
        print("test loss:", total_loss)  
        return total_loss

    def predict(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        folder_path = os.path.join(self.args.test_dir, setting)
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        test_data, test_loader = self._get_data(flag='test')
        random_idx = np.random.randint(0, int(len(test_data)/ dist.get_world_size()))
        criterion = self._select_criterion()
        criterion_step = nn.MSELoss(reduction='none')
        total_loss_step = []
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), total=len(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :-self.args.pred_len, :], dec_inp], dim=1).float()

                inference_steps = self.args.output_len // self.args.pred_len
                dis = self.args.output_len - inference_steps * self.args.pred_len
                if dis != 0:
                    inference_steps += 1
                
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        if self.args.covariate:
                            pred_vars_with_ground_truth_covariate = torch.cat(
                                [
                                    pred_y[-1], 
                                    batch_y[:, self.args.seq_len-self.args.input_len+(j-1)*self.args.pred_len:self.args.seq_len-self.args.input_len+j*self.args.pred_len, self.args.n_pred_vars:]
                                ],
                                dim=-1,
                            )
                            batch_x = torch.cat([batch_x[:, self.args.input_len:, :], pred_vars_with_ground_truth_covariate], dim=1)
                        else:
                            batch_x = torch.cat([batch_x[:, self.args.input_len:, :], pred_y[-1]], dim=1)
                        tmp = batch_y_mark[:, j - 1:j, :]
                        batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                    if self.args.output_attention:
                        # output used to calculate loss misaligned patch_len compared to input
                        outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)
                    else:
                        # only use the forecast window to calculate loss
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, n_pred_vars=self.args.n_pred_vars)

                    if self.args.covariate:
                        pred_y.append(outputs[:, -self.args.pred_len:, :self.args.n_pred_vars])
                    else:
                        pred_y.append(outputs[:, -self.args.pred_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-self.args.pred_len+dis, :]
                if self.args.covariate:
                    batch_y = batch_y[:, -self.args.output_len:, :self.args.n_pred_vars].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.output_len:, :].to(self.device)

                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()

                loss = criterion(outputs, batch_y)
                loss_step = criterion_step(outputs, batch_y).mean(dim=(0,2))
               
                loss_step = loss_step.detach().cpu()
                loss = loss.detach().cpu()
                total_loss_step.append(loss_step)
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])

                if i*test_loader.batch_size <= random_idx < (i+1)*test_loader.batch_size and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                    batch_ptr = random_idx % test_loader.batch_size
                    for feat_idx in range(outputs.shape[-1]):
                        pred_feat = outputs[batch_ptr, :, feat_idx].cpu().numpy()
                        true_feat = batch_y[batch_ptr, :, feat_idx].cpu().numpy()
                        plt.scatter(range(1,self.args.output_len+1), pred_feat, s=10, c= 'red',label="Pred")
                        plt.plot(range(1,self.args.output_len+1), true_feat, label="GT")
                        plt.title(f"Prediction and GroundTurth of Feature {feat_idx + 1}")
                        plt.xlabel("Prediction Step")
                        plt.ylabel("Elec value")
                        plt.legend()
                        plt.savefig(os.path.join(folder_path, f"feat_{feat_idx+1}_of_multi-step_test_data_{random_idx}th.png"))
                        plt.clf()

                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss_step = torch.tensor(np.average(total_loss_step, weights=total_count, axis=0))
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss_step = total_loss_step / dist.get_world_size()
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss_step = np.average(total_loss_step, weights=total_count, axis=0)
            total_loss = np.average(total_loss, weights=total_count)

        plt.scatter(range(1,self.args.output_len+1), total_loss_step, s=10)
        plt.title(f"loss in multi-step test set")
        plt.xlabel("Prediction Step")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig(os.path.join(folder_path, f"multi-step_test_data_mse.png"))
        plt.clf()     
        print("test loss:", total_loss)  
        return total_loss
    
    def prune(self, setting, train=0, prune_ratio=0.2, remove_mask=False):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        if train == 0:
            # 全局剪枝：在所有指定的 weight 张量里，按照 L1 范数最小的 20% 剪掉
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_ratio,
            )
            if remove_mask:
                # （可选）将 mask 应用到权重上，真正把剪掉的连接永久删除
                for module, name in parameters_to_prune:
                    prune.remove(module, name)
                print('Model parameters: ', sum(torch.count_nonzero(p).item() for p in self.model.parameters()))
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            steps = int(prune_ratio / 0.05)
            if steps % 0.05 != 0:
                last_prune_ratio = prune_ratio - (steps - 1) * 0.05
                steps += 1
            for i in range(steps):
                if i == steps - 1 and steps % 0.05 != 0:
                    step_prune_ratio = last_prune_ratio
                else:
                    step_prune_ratio = 0.05
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=step_prune_ratio,
                )
                self.finetune(setting)

            if remove_mask:
                # （可选）将 mask 应用到权重上，真正把剪掉的连接永久删除
                for module, name in parameters_to_prune:
                    prune.remove(module, name)
                print('Model parameters: ', sum(torch.count_nonzero(p).item() for p in self.model.parameters()))
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
    
    def visualize(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        folder_path = os.path.join(self.args.test_dir, setting)
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        test_data, test_loader = self._get_data(flag='test')
        random_idx = np.random.randint(0, int(len(test_data)/ dist.get_world_size()))
        x, y, x_mark, y_mark = test_data[random_idx]
        x = torch.tensor(x).reshape(1, -1, x.shape[-1]).float().to(self.device)
        y = torch.tensor(y).reshape(1, -1, y.shape[-1]).float().to(self.device)
        x_mark = torch.tensor(x_mark).reshape(1, -1, x_mark.shape[-1]).float().to(self.device)
        y_mark = torch.tensor(y_mark).reshape(1, -1, y_mark.shape[-1]).float().to(self.device)
        dec_inp = torch.zeros_like(y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([y[:, :-self.args.pred_len, :], dec_inp], dim=1).float()
        
        self.model.eval()
        with torch.no_grad():
            if self.args.output_attention:
                outputs, attns = self.model(x, x_mark, dec_inp, y_mark, n_pred_vars=self.args.n_pred_vars)
                
                # 计算patch数量
                n_patches = int(self.args.seq_len / self.args.patch_len)
                n_feat = x.shape[-1]
                
                # 绘制注意力热力图
                for layer_idx, layer_attn in enumerate(attns):
                    layer_attn = layer_attn.squeeze(0)  # [n_heads, n_patches*n_feat, n_patches*n_feat]
                    # 平均所有头的注意力权重
                    avg_attn = torch.mean(layer_attn, dim=0)  # [n_patches*n_feat, n_patches*n_feat]
                    attn_matrix = avg_attn.cpu().numpy()
                    
                    # 创建patch和特征的标签
                    patch_labels = []
                    for p in range(n_patches):
                        for f in range(n_feat):
                            patch_labels.append(f'P{p+1}F{f+1}')
                    
                    plt.figure(figsize=(15, 12))
                    sns.heatmap(attn_matrix, cmap='viridis',
                              xticklabels=patch_labels[::n_feat],  # 每隔n_feat个显示一个标签
                              yticklabels=patch_labels[::n_feat])
                    plt.title(f'Attention Heatmap - Layer {layer_idx + 1}')
                    plt.xlabel('Key Position (Patch-Feature)')
                    plt.ylabel('Query Position (Patch-Feature)')
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    plt.savefig(os.path.join(folder_path, f'attention_heatmap_layer_{layer_idx + 1}.png'))
                    plt.close()
                
                # 绘制时间-通道热力图
                # 计算每个patch对每个特征的注意力权重
                time_channel_attn = torch.zeros(n_patches, n_feat).to(self.device)
                for layer_attn in attns:
                    layer_attn = layer_attn.squeeze(0)  # [n_heads, n_patches*n_feat, n_patches*n_feat]
                    # 平均所有头的注意力权重
                    avg_attn = torch.mean(layer_attn, dim=0)  # [n_patches*n_feat, n_patches*n_feat]
                    
                    # 对每个patch，计算其对所有特征的注意力
                    for p in range(n_patches):
                        patch_start = p * n_feat
                        patch_end = (p + 1) * n_feat
                        # 计算当前patch对所有特征的注意力
                        patch_attn = avg_attn[patch_start:patch_end].mean(dim=0)  # [n_patches*n_feat]
                        # 将注意力权重重塑为[n_patches, n_feat]并累加
                        patch_attn = patch_attn.reshape(n_patches, n_feat)
                        time_channel_attn += patch_attn
                
                time_channel_attn = time_channel_attn.cpu().numpy()
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(time_channel_attn, cmap='viridis',
                           xticklabels=[f'Var{i+1}' for i in range(n_feat)],
                           yticklabels=[f'Patch{i+1}' for i in range(n_patches)])
                plt.title('Time-Channel Attention Heatmap')
                plt.xlabel('Features')
                plt.ylabel('Patches')
                plt.tight_layout()
                plt.savefig(os.path.join(folder_path, 'time_channel_heatmap.png'))
                plt.close()
                
            else:
                outputs = self.model(x, x_mark, dec_inp, y_mark, n_pred_vars=self.args.n_pred_vars)

            if self.args.covariate:
                outputs = outputs[:, :, :self.args.n_pred_vars]
                y = y[:, :, :self.args.n_pred_vars]

            # 绘制预测结果
            for feat_idx in range(outputs.shape[-1]):
                pred_feat = outputs[:, -self.args.pred_len:, feat_idx].cpu().numpy()
                true_feat = y[:, -self.args.pred_len:, feat_idx].cpu().numpy()
                if self.args.covariate and feat_idx >=self.args.n_pred_vars:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, self.args.pred_len+1), true_feat[0], label="GT")
                    plt.title(f"Ground Truth of Feature {feat_idx + 1}")
                    plt.xlabel("Time Step")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.savefig(os.path.join(folder_path, f"feat_{feat_idx+1}_of_test_data_{random_idx}th.png"))
                    plt.close()
                else:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(range(1, self.args.pred_len+1), pred_feat[0], s=10, c='red', label="Pred")
                    plt.plot(range(1, self.args.pred_len+1), true_feat[0], label="GT")
                    plt.title(f"Prediction and Ground Truth of Feature {feat_idx + 1}")
                    plt.xlabel("Prediction Step")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.savefig(os.path.join(folder_path, f"feat_{feat_idx+1}_of_test_data_{random_idx}th.png"))
                    plt.close()

