import os
import time
import pandas as pd
import warnings
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
import torch.multiprocessing
from torch.nn.utils import prune
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
torch.multiprocessing.set_sharing_strategy('file_system')

from data_provider.data_factory import data_provider, concat_data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):        
        # rec_token_count代表这模型生成有几个patch是有效的
        super(Exp_Anomaly_Detection, self).__init__(args)
        if self.args.use_ims:
            rec_token_count = (self.args.seq_len - 2 * self.args.patch_len) // self.args.patch_len
        else:
            rec_token_count = self.args.seq_len // self.args.patch_len
        self.rec_token_count = rec_token_count

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.freeze_layer:
            self._freeze_model(model)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
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
        if self.args.model == 'Timer_multivariate' and self.args.data == 'multivariate_anomaly':
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch_x in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                if self.args.use_ims:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                    # input and output are completely aligned
                    outputs = self.model(batch_x, None, None, None)

                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_x = batch_x[:, :, :self.args.n_pred_vars]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def finetune(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if self.args.train_test:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        print(f"train steps per epoch: {train_steps}")
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_opacus:
            privacy_engine = PrivacyEngine()
            self.model, model_optim, train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=model_optim,
                data_loader=train_loader,
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
        )

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch_x in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                if self.args.use_ims:
                    # backward overlapping parts between outputs and inputs
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:, :]
                else:
                    # input and output are completely aligned
                    outputs = self.model(batch_x, None, None, None)
                
                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_x = batch_x[:, :, :self.args.n_pred_vars]

                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            if self.args.use_opacus:
                epsilon, alpha = privacy_engine.get_privacy_spent(delta=1e-5)
                print(f"Epoch {epoch+1}: cost time: {time.time() - epoch_time:.2f}s, loss={loss.item():.4f}, ε={epsilon:.2f}, α={alpha}")
            else:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, "checkpoint.pth")
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def _gen_manual_anomaly_patch(self, k=5):
        if hasattr(self, 'anomaly_dataset') and hasattr(self, 'anomaly_dataloader'):
            return self.anomaly_dataset, self.anomaly_dataloader
        if not hasattr(self, 'test_dataset') or not hasattr(self, 'test_dataloader'):
            test_data, test_loader = self._get_data(flag='test')
        manual_anomaly_patch_index_list = random.choices(range(len(self.test_dataset) * self.rec_token_count), k=k) # 对应预测patch的起始index为0
        self.manual_anomaly_patch_index_list = manual_anomaly_patch_index_list
        anomalized_batches = []
        for i, orginal_batch_x in enumerate(self.test_dataloader):
            batch_x = orginal_batch_x.clone().float().cpu()
            # manual modify to create anomaly
            valid_manual_anomaly_patch_index_list = [patch_idx for patch_idx in manual_anomaly_patch_index_list if i*self.rec_token_count <= patch_idx < (i+1)*self.rec_token_count]
            for patch_idx in valid_manual_anomaly_patch_index_list:
                token_start = (patch_idx % self.rec_token_count + 1) * self.args.patch_len
                token_end = token_start + self.args.patch_len
                # modify std to 5 times
                batch_x_std = (batch_x[:, token_start:token_end, :self.args.n_pred_vars] - self.test_dataset.mean_vector[:self.args.n_pred_vars]) / self.test_dataset.std_vector[:self.args.n_pred_vars]
                mask_neg = (batch_x_std < 0) & (batch_x_std > -1)
                mask_pos = (batch_x_std >= 0) & (batch_x_std < 0)
                batch_x_std[mask_neg] = -1.0
                batch_x_std[mask_pos] = 1.0
                batch_x[:, token_start:token_end, :self.args.n_pred_vars] += 4* batch_x_std * self.test_dataset.std_vector[:self.args.n_pred_vars]
            anomalized_batches.append(batch_x.detach().to('cpu'))

        anomalized_X = torch.cat(anomalized_batches, dim=0)  # [N, seq_len, D]
        anomaly_dataset = TensorDataset(anomalized_X)

        # 沿用原 test_dataloader 的常用参数（如取不到则给出保守默认）
        batch_size  = getattr(self.test_dataloader, 'batch_size', 64)
        num_workers = getattr(self.test_dataloader, 'num_workers', 0)
        pin_memory  = getattr(self.test_dataloader, 'pin_memory', False)
        drop_last   = getattr(self.test_dataloader, 'drop_last', False)

        anomaly_loader = DataLoader(
            anomaly_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        # 如需在对象内保存，附上：
        self.anomaly_dataset = anomaly_dataset
        self.anomaly_dataloader = anomaly_loader
        return anomaly_dataset, anomaly_loader

    def test(self, setting, test=0):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        test_data, test_loader = self._get_data(flag='test')
        score_list = []
        folder_path = os.path.join(self.args.test_dir, setting, os.path.splitext(self.args.data_path)[0])
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # border_start = self.find_border_number(self.args.data_path)
        # border1, border2 = self.find_border(self.args.data_path)

        token_count = 0
        rec_token_count = self.rec_token_count
        
        input_list = []
        output_list = []
        anomaly_dataset, anomaly_loader = self._gen_manual_anomaly_patch(k=5)
        # manual_anomaly_patch_index_list = random.choices(range(len(test_data) * rec_token_count), k=5) # 对应预测patch的起始index为0
        anomaly_input_list = []
        with torch.no_grad():
            for i, (orginal_batch_x, batch_x) in enumerate(zip(test_loader, anomaly_loader)):
                batch_x = batch_x[0]
                # orginal_batch_x = batch_x.clone().float().to(self.device)
                # # manual modify to create anomaly
                # valid_manual_anomaly_patch_index_list = [patch_idx for patch_idx in manual_anomaly_patch_index_list if i*rec_token_count <= patch_idx < (i+1)*rec_token_count]
                # for patch_idx in valid_manual_anomaly_patch_index_list:
                #     token_start = (patch_idx % rec_token_count + 1) * self.args.patch_len
                #     token_end = token_start + self.args.patch_len
                #     # modify std to 5 times
                #     batch_x_std = (batch_x[:, token_start:token_end, :self.args.n_pred_vars] - test_data.mean_vector[:self.args.n_pred_vars]) / test_data.std_vector[:self.args.n_pred_vars]
                #     mask_neg = (batch_x_std < 0) & (batch_x_std > -1)
                #     mask_pos = (batch_x_std >= 0) & (batch_x_std < 0)
                #     batch_x_std[mask_neg] = -1.0
                #     batch_x_std[mask_pos] = 1.0
                #     batch_x[:, token_start:token_end, :self.args.n_pred_vars] += 4* batch_x_std * test_data.std_vector[:self.args.n_pred_vars]

                batch_x = batch_x.float().to(self.device)
                # reconstruct the input sequence and record the loss as a sorted list
                if self.args.use_ims:
                    outputs = self.model(batch_x[:, :-self.args.patch_len, :], None, None, None)
                    batch_x = batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                    orginal_batch_x = orginal_batch_x[:, self.args.patch_len:-self.args.patch_len, :]
                    outputs = outputs[:, :-self.args.patch_len, :]
                else:
                    outputs = self.model(batch_x, None, None, None)

                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_x = batch_x[:, :, :self.args.n_pred_vars]
                    orginal_batch_x = orginal_batch_x[:, :, :self.args.n_pred_vars]

                n_feats = batch_x.shape[-1]
                input_list.append(orginal_batch_x.reshape(-1, n_feats).detach().cpu().numpy())
                anomaly_input_list.append(batch_x.reshape(-1, n_feats).detach().cpu().numpy())
                output_list.append(outputs.reshape(-1, n_feats).detach().cpu().numpy())
                for j in range(rec_token_count):
                    # criterion
                    token_start = j * self.args.patch_len
                    token_end = token_start + self.args.patch_len
                    for feat_idx in range(outputs.shape[-1]):
                        pred_feat_norm = (outputs[:, token_start:token_end, feat_idx] - test_data.mean_vector[feat_idx]) / test_data.std_vector[feat_idx]
                        true_feat_norm = (batch_x[:, token_start:token_end, feat_idx] - test_data.mean_vector[feat_idx]) / test_data.std_vector[feat_idx]
                        score = torch.mean(self.anomaly_criterion(true_feat_norm, pred_feat_norm), dim=-1)
                        score = score.detach().cpu().numpy()
                        score = np.mean(score)
                        score_list.append(((token_count, feat_idx), score))
                    token_count += 1

        # 每个patch都会有一个赋分，按照赋分进行排序
        score_list.sort(key=lambda x: x[1], reverse=True)
        # 将完整的测试数据的异常点标注出来，其中为保证展示的完整，向前后各拓展半个patch
        input = np.concatenate(input_list, axis=0).reshape(-1, n_feats)
        anomaly_input = np.concatenate(anomaly_input_list, axis=0).reshape(-1, n_feats)
        output = np.concatenate(output_list, axis=0).reshape(-1, n_feats)

        anomaly_list, anomaly_index_list = [], []
        for i, ((patch_idx, feat_idx), score) in enumerate(score_list):
            if self.args.loss_threshold <= score:
                anomaly_index_list.append((patch_idx*self.args.patch_len+1, (patch_idx+1)*self.args.patch_len, feat_idx))
                if len(test_data.time_list) != 0:
                    patch_start_time = test_data.time_list[(patch_idx+1)*self.args.patch_len]
                    patch_end_time = test_data.time_list[(patch_idx+2)*self.args.patch_len-1]
                    anomaly_list.append((patch_start_time, patch_end_time, feat_idx))
                else:
                    anomaly_list.append((patch_idx, feat_idx))
        if len(test_data.time_list) != 0:
            df = pd.DataFrame(anomaly_list, columns=['start_time', 'end_time', 'feat_idx'])
        else:
            df = pd.DataFrame(anomaly_list, columns=['patch_idx', 'feat_idx'])
        df.to_csv(os.path.join(folder_path, "anomaly_result.csv"),index=False)

        for feat_idx in range(self.args.n_pred_vars):
            input_feat = input[:, feat_idx]
            anomaly_input_feat = anomaly_input[:, feat_idx]
            output_feat = output[:, feat_idx]
            length = len(input_feat)

            plt.scatter(range(1,length+1), output_feat, s=10, c= 'red',label="Pred")
            plt.plot(range(1,length+1), input_feat, label="GT", c='black')
            plt.plot(range(1,length+1), anomaly_input_feat, label="Anomaly_GT", c='blue')

            for xmin, xmax, anomaly_feat_idx in anomaly_index_list:
                if anomaly_feat_idx == feat_idx:
                    plt.axvspan(xmin, xmax, ymin=0, ymax=1, facecolor='yellow', alpha=0.5, edgecolor='none')
            
            for patch_idx in self.manual_anomaly_patch_index_list:
                plt.axvspan(patch_idx*self.args.patch_len+1, (patch_idx+1)*self.args.patch_len, ymin=0, ymax=1, facecolor='cyan', alpha=0.5, edgecolor='none')


            plt.title(f"Prediction and GroundTurth of Feature {feat_idx + 1}")
            plt.xlabel("Prediction Step")
            plt.ylabel("Elec value")
            plt.legend()
            plt.savefig(os.path.join(folder_path, f"feat_{feat_idx+1}_of_test_data_{os.path.splitext(self.args.data_path)}.png"))
            plt.clf()

        return df

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