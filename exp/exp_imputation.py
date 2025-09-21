import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import prune
from torch.utils.data import TensorDataset, DataLoader
from opacus import PrivacyEngine
from data_provider.data_factory import data_provider, concat_data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.freeze_layer:
            model = self._freeze_model(model)
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
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        mask_dataset, mask_loader = self._mask_data(flag='val')
        with torch.no_grad():
            for i, ((batch_x, batch_y, batch_x_mark, batch_y_mark), (inp, mask)) in enumerate(zip(vali_loader, mask_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                inp = inp.float().to(self.device)
                mask = mask.float().to(self.device)

                # random mask
                batch_size, seq_len, n_feats = batch_x.shape
                assert seq_len % self.args.patch_len == 0
                # mask = torch.rand((batch_size, seq_len // self.args.patch_len, n_feats)).to(self.device)
                # mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                # mask[mask <= self.args.mask_rate] = 0  # masked
                # mask[mask > self.args.mask_rate] = 1  # remained
                # mask = mask.view(mask.size(0), -1, mask.size(-1))
                # mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                # inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask=mask, n_pred_vars=self.args.n_pred_vars)

                # 如果use_ims则认为output为patch_len右移的seq_len,否则认为没有产生时间步移动
                if self.args.use_ims:
                    outputs = outputs[:, :-self.args.patch_len, :]
                else:
                    outputs = outputs[:, self.args.patch_len:, :]
                
                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_x = batch_x[:, :, :self.args.n_pred_vars]
                    mask = mask[:, :, :self.args.n_pred_vars]

                pred = outputs.detach().cpu()
                true = batch_x[:, self.args.patch_len:, :].detach().cpu()
                mask = mask[:, self.args.patch_len:, :].detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def _mask_data(self, flag, test_version="test"):
        if flag == 'train':
            if hasattr(self, 'mask_train_dataset') and hasattr(self, 'mask_train_dataloader'):
                return self.mask_train_dataset, self.mask_train_dataloader
            if not hasattr(self, 'train_dataset') or not hasattr(self, 'train_dataloader'):
                train_data, train_loader = self._get_data(flag='train')
            data_loader = self.train_dataloader
        if flag == 'val':
            if hasattr(self, 'mask_val_dataset') and hasattr(self, 'mask_val_dataloader'):
                return self.mask_val_dataset, self.mask_val_dataloader
            if not hasattr(self, 'val_dataset') or not hasattr(self, 'val_dataloader'):
                vali_data, vali_loader = self._get_data(flag='val')
            data_loader = self.val_dataloader
        if flag == 'test':
            if hasattr(self, 'mask_test_dataset') and hasattr(self, 'mask_test_dataloader'):
                return self.mask_test_dataset, self.mask_test_dataloader
            if not hasattr(self, 'test_dataset') or not hasattr(self, 'test_dataloader'):
                test_data, test_loader = self._get_data(flag='test')
            data_loader = self.test_dataloader
        
        inp_patches, mask_patches = [], []
        for i, (original_batch_x, _, _, _) in enumerate(data_loader):
            batch_x = original_batch_x.clone().float().cpu()

            # random mask
            batch_size, seq_len, n_feats = batch_x.shape
            assert seq_len % self.args.patch_len == 0

            if test_version == "test":
                mask = torch.rand((batch_size, seq_len // self.args.patch_len, n_feats))
                mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1) # [batch_size, seq_len // self.args.patch_len, self.args.patch_len, n_feats]
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.view(mask.size(0), -1, mask.size(-1)) # [batch_size, seq_len, n_feats] 如果一个patch内某个feat的mask为0，则所有点该feat均被mask为0
                mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                inp = batch_x.masked_fill(mask == 0, 0)
            elif test_version == "auto_process":
                # 检查 batch_x 中的 nan，若某 patch 某特征有 nan，则该 patch 该特征全为 0
                mask = torch.ones((batch_size, seq_len // self.args.patch_len, n_feats))
                batch_x_reshaped = batch_x.view(batch_size, seq_len // self.args.patch_len, self.args.patch_len, n_feats)
                nan_mask = torch.isnan(batch_x_reshaped).any(dim=2)  # [batch_size, n_patch, n_feats]
                mask[nan_mask] = 0
                mask = mask.repeat(1, self.args.patch_len, 1) # [batch_size, seq_len, n_feats]
                mask[:, :self.args.patch_len, :] = 1
                inp = batch_x.masked_fill(mask == 0, 0)

            inp_patches.append(inp.detach().to('cpu'))
            mask_patches.append(mask.detach().to('cpu'))

        inp_X = torch.cat(inp_patches, dim=0)  # [N, seq_len, D]
        mask_X = torch.cat(mask_patches, dim=0)  # [N, seq_len, D]
        mask_dataset = TensorDataset(inp_X, mask_X)

        # 沿用原 test_dataloader 的常用参数（如取不到则给出保守默认）
        batch_size  = getattr(data_loader, 'batch_size', 64)
        num_workers = getattr(data_loader, 'num_workers', 0)
        pin_memory  = getattr(data_loader, 'pin_memory', False)
        drop_last   = getattr(data_loader, 'drop_last', False)

        mask_loader = DataLoader(
            mask_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

        if flag == 'train':
            self.mask_train_dataset = mask_dataset
            self.mask_train_dataloader = mask_loader
        if flag == 'val':
            self.mask_val_dataset = mask_dataset
            self.mask_val_dataloader = mask_loader
        if flag == 'test':
            self.mask_test_dataset = mask_dataset
            self.mask_test_dataloader = mask_loader
        return mask_dataset, mask_loader

    def finetune(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        if self.args.train_test:
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting, f"subset{self.args.subset_rand_ratio}_mask{self.args.mask_rate}")
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        train_len = train_data.__len__() // self.args.batch_size * self.args.subset_rand_ratio
        print(f"train steps per epoch: {train_len}")

        if self.args.use_opacus:
            privacy_engine = PrivacyEngine()
            self.model, model_optim, train_loader = privacy_engine.make_private(
                module=self.model,
                optimizer=model_optim,
                data_loader=train_loader,
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
        )

        mask_dataset, mask_loader = self._mask_data(flag='train')
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, ((batch_x, batch_y, batch_x_mark, batch_y_mark), (inp, mask)) in enumerate(zip(train_loader, mask_loader)):
                if i > train_len:
                    break
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                inp = inp.float().to(self.device)
                mask = mask.float().to(self.device)
                # random mask
                batch_size, seq_len, n_feats = batch_x.shape
                assert seq_len % self.args.patch_len == 0
                # mask = torch.rand((batch_size, seq_len // self.args.patch_len, n_feats)).to(self.device)
                # mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1) # [batch_size, seq_len // self.args.patch_len, self.args.patch_len, n_feats]
                # mask[mask <= self.args.mask_rate] = 0  # masked
                # mask[mask > self.args.mask_rate] = 1  # remained
                # mask = mask.view(mask.size(0), -1, mask.size(-1)) # [batch_size, seq_len, n_feats] 如果一个patch内某个feat的mask为0，则所有点该feat均被mask为0
                # mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                # inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask=mask, n_pred_vars=self.args.n_pred_vars)

                # 如果use_ims则认为output为patch_len右移的seq_len,否则认为没有产生时间步移动
                batch_x = batch_x[:, self.args.patch_len:, :]
                mask = mask[:, self.args.patch_len:, :]
                if self.args.use_ims:
                    outputs = outputs[:, :-self.args.patch_len, :]
                else:
                    outputs = outputs[:, self.args.patch_len:, :]
                
                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_x = batch_x[:, :, :self.args.n_pred_vars]
                    mask = mask[:, :, :self.args.n_pred_vars]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
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
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if self.args.test_version == "test":
            folder_path = os.path.join(self.args.test_dir, setting, f"subset{self.args.subset_rand_ratio}_mask{self.args.mask_rate}")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mask_dataset, mask_loader = self._mask_data(flag='test')
        
        elif self.args.test_version == "auto_process":
            folder_path = os.path.join(self.args.test_dir, setting, f"subset{self.args.subset_rand_ratio}")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            mask_dataset, mask_loader = self._mask_data(flag='test', test_version="auto_process")

        else:
            raise ValueError("Invalid test_version. Supported versions: 'test', 'auto_process'.")
        
        random_indices = np.random.choice(range(len(test_data)), size=10, replace=False)
        preds = []
        trues = []
        masks = []
        
        self.model.eval()
        with torch.no_grad():
            for i, ((batch_x, batch_y, batch_x_mark, batch_y_mark), (inp, mask)) in tqdm(enumerate(zip(test_loader, mask_loader)), total=len(test_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                inp = inp.float().to(self.device)
                mask = mask.float().to(self.device)

                # random mask
                batch_size, seq_len, n_feats = batch_x.shape
                assert seq_len % self.args.patch_len == 0
                # # mask = torch.rand((batch_size, seq_len // self.args.patch_len, n_feats)).to(self.device)
                # mask = torch.cat([torch.rand((batch_size, seq_len // self.args.patch_len, self.args.n_pred_vars)), torch.ones((batch_size, seq_len // self.args.patch_len, n_feats - self.args.n_pred_vars))], dim=-1).to(self.device)
                # mask = mask.unsqueeze(2).repeat(1, 1, self.args.patch_len, 1)
                # mask[mask <= self.args.mask_rate] = 0  # masked
                # mask[mask > self.args.mask_rate] = 1  # remained
                # mask = mask.view(mask.size(0), -1, mask.size(-1))
                # mask[:, :self.args.patch_len, :] = 1  # first patch is always observed
                # inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask=mask, n_pred_vars=self.args.n_pred_vars)

                # eval
                # 如果use_ims则认为output为patch_len右移的seq_len,否则认为没有产生时间步移动
                if self.args.use_ims:
                    outputs = outputs[:, :-self.args.patch_len, :]
                else:
                    outputs = outputs[:, self.args.patch_len:, :]
                
                if self.args.covariate:
                    # 假设如果是covariate任务，我的batch_y仅包含pred_vars的值，即最后一个dim为self.args.n_pred_vars
                    outputs = outputs[:, :, :self.args.n_pred_vars]
                    batch_x = batch_x[:, :, :self.args.n_pred_vars]
                    mask = mask[:, :, :self.args.n_pred_vars]

                pred = outputs.detach().cpu().numpy()
                true = batch_x[:, self.args.patch_len:, :].detach().cpu().numpy()
                test_mask = mask[:, self.args.patch_len:, :].detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(test_mask)

                valid_indices = [idx for idx in random_indices if i*test_loader.batch_size <= idx < (i+1)*test_loader.batch_size]
                for random_idx in valid_indices:
                    batch_ptr = random_idx % test_loader.batch_size
                    for feat_idx in range(outputs.shape[-1]):
                        true_feat = np.concatenate((batch_x[batch_ptr, :self.args.patch_len, feat_idx].detach().cpu().numpy(), true[batch_ptr, :, feat_idx]), axis=0)
                        pred_feat = np.concatenate((batch_x[batch_ptr, :self.args.patch_len, feat_idx].detach().cpu().numpy(), pred[batch_ptr, :, feat_idx]), axis=0)
                        mask_feat = mask[batch_ptr, :, feat_idx].detach().cpu().numpy()
                        pred_feat_with_gt_filled = true_feat * mask_feat + pred_feat * (1 - mask_feat)
                        
                        plt.plot(range(1,seq_len+1), pred_feat_with_gt_filled, c= 'red',label="Pred")
                        plt.plot(range(1,seq_len+1), true_feat, label="GT")

                        intervals = []
                        in_interval = False
                        start = None
                        for i in range(seq_len):
                            if mask_feat[i] == 0 and not in_interval:
                                in_interval = True
                                start = i + 1
                            if mask_feat[i] and in_interval:
                                end = i
                                intervals.append((start, end))
                                in_interval = False
                        if in_interval:
                            intervals.append((start, seq_len))
                        for (xmin, xmax) in intervals:
                            plt.axvspan(xmin, xmax, ymin=0, ymax=1, facecolor='orange', alpha=0.5, edgecolor='none')
    
                        plt.title(f"Prediction and GroundTurth of Feature {feat_idx + 1}")
                        plt.xlabel("Prediction Step")
                        plt.ylabel("Elec value")
                        plt.legend()
                        plt.savefig(os.path.join(folder_path, f"feat_{feat_idx+1}_of_test_data_{random_idx}th.png"))
                        plt.clf()

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)

        if self.args.test_version == "test":
            # result save
            mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
            print('mse:{}, mae:{}'.format(mse, mae))
            f = open(os.path.join(folder_path, "result_imputation.txt"), 'a')
            f.write(setting + "  \n")
            f.write('mse:{}, mae:{}'.format(mse, mae))
            f.write('\n')
            f.write('\n')
            f.close()

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        
        elif self.args.test_version == "auto_process":
            np.save(folder_path + 'pred.npy', preds)
        
        else:
            raise ValueError("Invalid test_version. Supported versions: 'test', 'auto_process'.")
        
        return

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