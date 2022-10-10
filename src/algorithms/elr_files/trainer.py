import warnings
from typing import TypeVar, List, Tuple
import torch
from tqdm import tqdm
from abc import abstractmethod
from numpy import inf
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model1, model2, model_ema1, model_ema2, train_criterion1,
                 train_criterion2, metrics, optimizer1, optimizer2,  val_criterion,
                 model_ema1_copy, model_ema2_copy, epochs, save_period, monitor,early_stop,save_dir):


        # setup GPU device if available, move model into configured device
        self.device, self.device_ids = self._prepare_device(1)

        if len(self.device_ids) > 1:
            print('Using Multi-Processing!')

        self.model1 = model1.to(self.device + str(self.device_ids[0]))
        self.model2 = model2.to(self.device + str(self.device_ids[-1]))

        if model_ema1 is not None:
            self.model_ema1 = model_ema1.to(self.device + str(self.device_ids[0]))
            self.model_ema2_copy = model_ema2_copy.to(self.device + str(self.device_ids[0]))
        else:
            self.model_ema1 = None
            self.model_ema2_copy = None

        if model_ema2 is not None:
            self.model_ema2 = model_ema2.to(self.device + str(self.device_ids[-1]))
            self.model_ema1_copy = model_ema1_copy.to(self.device + str(self.device_ids[-1]))
        else:
            self.model_ema2 = None
            self.model_ema1_copy = None

        if self.model_ema1 is not None:
            for param in self.model_ema1.parameters():
                param.detach_()

            for param in self.model_ema2_copy.parameters():
                param.detach_()

        if self.model_ema2 is not None:
            for param in self.model_ema2.parameters():
                param.detach_()

            for param in self.model_ema1_copy.parameters():
                param.detach_()

        self.train_criterion1 = train_criterion1.to(self.device + str(self.device_ids[0]))
        self.train_criterion2 = train_criterion2.to(self.device + str(self.device_ids[-1]))

        self.val_criterion = val_criterion

        self.metrics = metrics

        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2

        self.epochs = epochs
        self.save_period = save_period
        self.monitor = monitor

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = early_stop

        self.start_epoch = 1

        self.global_step = 0

        self.checkpoint_dir = save_dir

        # params
        self.warmup = 0



    @abstractmethod
    def _train_epoch(self,epoch, model, model_ema, model_ema2, data_loader, train_criterion, optimizer, lr_scheduler, device = 'cpu', queue = None):
        """
        Training logic for an epoch

        :param epoch: Current epochs number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """



        not_improved_count = 0

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
            if epoch <= self.warmup:
                continue


            else:
                if len(self.device_ids) > 1:

                    continue
                else:
                    result1 = self._train_epoch(epoch, self.model1, self.model_ema1, self.model_ema2, self.data_loader1,
                                                self.train_criterion1, self.optimizer1, self.lr_scheduler1,
                                                self.device + str(self.device_ids[0]))
                    result2 = self._train_epoch(epoch, self.model2, self.model_ema2, self.model_ema1, self.data_loader2,
                                                self.train_criterion2, self.optimizer2, self.lr_scheduler2,
                                                self.device + str(self.device_ids[-1]))

                self.global_step += result1['local_step']
                if len(self.device_ids) > 1:

                    continue
                else:
                    if self.do_validation:
                        val_log = self._valid_epoch(epoch, self.model1, self.model2,
                                                    self.device + str(self.device_ids[0]))
                        result1.update(val_log)
                        result2.update(val_log)
                    if self.do_test:
                        test_log = self._test_epoch(epoch, self.model1, self.model2,
                                                    self.device + str(self.device_ids[0]))
                        result1.update(test_log)
                        result2.update(test_log)

                        # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result1.items():
                if key == 'metrics':
                    log.update({'Net1' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                    log.update({'Net2' + mtr.__name__: result2[key][i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'test_metrics':
                    log.update({'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log['Net1' + key] = value
                    log['Net2' + key] = result2[key]

            # print logged informations to the screen
            for key, value in log.items():
                print('    {:15s}: {}'.format(str(key), value))
                # self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = 'cuda:'  # torch.device('cuda:' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model1).__name__

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict1': self.model1.state_dict(),
            'state_dict2': self.model2.state_dict(),
            'optimizer1': self.optimizer1.state_dict(),
            'optimizer2': self.optimizer2.state_dict(),
            'monitor_best': self.mnt_best
            # 'config': self.config
        }
        filename = str(self.checkpoint_dir + '/checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir +'/model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth at: {} ...".format(best_path))



class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model1, model2, model_ema1, model_ema2, train_criterion1, train_criterion2, metrics, optimizer1, optimizer2,
                 data_loader1, data_loader2,
                 valid_data_loader,
                 test_data_loader,
                 lr_scheduler1, lr_scheduler2,
                  val_criterion,
                 model_ema1_copy, model_ema2_copy,epochs, save_period, monitor,early_stop,save_dir,num_classes,ema_step,mixup_alpha,ema_alpha,ema_update):
        super().__init__(model1, model2, model_ema1, model_ema2, train_criterion1, train_criterion2, 
                         metrics, optimizer1, optimizer2, val_criterion, model_ema1_copy, model_ema2_copy,epochs, save_period, monitor,early_stop,save_dir)
        # self.config = config.config
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2
        self.num_classes = num_classes
        self.ema_step = ema_step
        self.mixup_alpha=mixup_alpha
        self.ema_alpha = ema_alpha
        self.ema_update = ema_update
        self.len_epoch = len(self.data_loader1)
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_test = self.test_data_loader is not None
        self.lr_scheduler1 = lr_scheduler1
        self.lr_scheduler2 = lr_scheduler2
        self.log_step = int(np.sqrt(self.data_loader1.batch_size))
        self.train_loss_list: List[float] = []
        self.val_loss_list: List[float] = []
        self.test_loss_list: List[float] = []
        

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)

        return acc_metrics

    def _train_epoch(self, epoch, model, model_ema, model_ema2, data_loader, train_criterion, optimizer, lr_scheduler, device = 'cpu', queue = None):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        model.train()
        model_ema.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        total_metrics_ema = np.zeros(len(self.metrics))

        if hasattr(data_loader.dataset, 'num_raw_example'):
            num_examp = data_loader.dataset.num_raw_example
        else:
            num_examp = len(data_loader.dataset)

        local_step = 0



        with tqdm(data_loader) as progress:
            for batch_idx, (data, target, indexs, _) in enumerate(progress):
                progress.set_description_str(f'Train epoch {epoch}')

                data_original = data
                target_original = target

                target = torch.zeros(len(target), self.num_classes).scatter_(1, target.view(-1,1), 1)
                data, target, target_original = data.to(device), target.float().to(device), target_original.to(device)
                
                data, target, mixup_l, mix_index = self._mixup_data(data, target,  alpha = self.mixup_alpha, device = device)
                
                output = model(data)

                data_original = data_original.to(device)
                output_original  = model_ema2(data_original)
                output_original = output_original.data.detach()
                train_criterion.update_hist(epoch, output_original, indexs.numpy().tolist(), mix_index = mix_index, mixup_l = mixup_l)
                
                local_step += 1
                loss, probs = train_criterion(self.global_step + local_step, output, target)
                
                optimizer.zero_grad()
                loss.backward() 

                
                optimizer.step()
                
                self.update_ema_variables(model, model_ema, self.global_step + local_step, self.ema_alpha)


                self.train_loss_list.append(loss.item())
                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, target.argmax(dim=1))
                if output_original is not None:
                    total_metrics_ema += self._eval_metrics(output_original, target.argmax(dim=1))


                if batch_idx % self.log_step == 0:
                    progress.set_postfix_str(' {} Loss: {:.6f}'.format(
                        self._progress(batch_idx),
                        loss.item()))

                if batch_idx == self.len_epoch:
                    break

        if hasattr(data_loader, 'run'):
            data_loader.run()


        log = {
            'global step': self.global_step,
            'local_step': local_step,
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
            'metrics_ema': (total_metrics_ema / self.len_epoch).tolist(),
            'learning rate': lr_scheduler.get_lr()
        }


        if lr_scheduler is not None:
            lr_scheduler.step()

        if queue is None:
            return log
        else:
            queue.put(log)


    def _valid_epoch(self, epoch, model1, model2, device = 'cpu', queue = None):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        model1.eval()
        model2.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.valid_data_loader) as progress:
                for batch_idx, (data, target, _, _) in enumerate(progress):
                    progress.set_description_str(f'Valid epoch {epoch}')
                    data, target = data.to(device), target.to(device)
                    
                    output1 = model1(data)
                    output2 = model2(data)

                    output = 0.5*(output1 + output2)

                    loss = self.val_criterion(output, target)


                    self.val_loss_list.append(loss.item())
                    total_val_loss += loss.item()
                    total_val_metrics += self._eval_metrics(output, target)

        if queue is None:
            return {
                'val_loss': total_val_loss / len(self.valid_data_loader),
                'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
            }
        else:
            queue.put({
                'val_loss': total_val_loss / len(self.valid_data_loader),
                'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
            })

    def _test_epoch(self, epoch, model1, model2, device = 'cpu', queue = None):
        """
        Test after training an epoch

        :return: A log that contains information about test

        Note:
            The Test metrics in log must have the key 'val_metrics'.
        """
        model1.eval()
        model2.eval()

        total_test_loss = 0
        total_test_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            with tqdm(self.test_data_loader) as progress:
                for batch_idx, (data, target,indexs,_) in enumerate(progress):
                    progress.set_description_str(f'Test epoch {epoch}')
                    data, target = data.to(device), target.to(device)

                    output1 = model1(data)
                    output2 = model2(data)
                    
                    output = 0.5*(output1 + output2)
                    loss = self.val_criterion(output, target)

                    self.test_loss_list.append(loss.item())
                    total_test_loss += loss.item()
                    total_test_metrics += self._eval_metrics(output, target)
            

        #add histogram of model parameters to the tensorboard
        if queue is None:
            return {
                'test_loss': total_test_loss / len(self.test_data_loader),
                'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
            }
        else:
            queue.put({
                'test_loss': total_test_loss / len(self.test_data_loader),
                'test_metrics': (total_test_metrics / len(self.test_data_loader)).tolist()
            })


    def update_ema_variables(self, model, model_ema, global_step, alpha_=0.997):
        # Use the true average until the exponential average is more correct
        if alpha_ == 0:
            ema_param.data = param.data
        else:
            if self.ema_update:
                alpha = self.sigmoid_rampup(global_step + 1, self.ema_step)*alpha_
            else:
                alpha = min(1 - 1 / (global_step + 1), alpha_)
            for ema_param, param in zip(model_ema.parameters(), model.parameters()):
                ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def sigmoid_rampup(self,current, rampup_length):
        """Exponential rampup from  2"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader1, 'n_samples'):
            current = batch_idx * self.data_loader1.batch_size
            total = self.data_loader1.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _mixup_data(self, x, y, alpha=1.0,  device = 'cpu'):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1-lam)
            batch_size = x.size()[0]
            mix_index = torch.randperm(batch_size).to(device)

            mixed_x = lam * x + (1 - lam) * x[mix_index, :]#
            mixed_target = lam * y + (1 - lam) * y[mix_index, :]


            return mixed_x, mixed_target, lam, mix_index
        else:
            lam = 1
            return x, y, lam, ...


    def _mixup_criterion(self, pred, y_a, y_b, lam, *args):
        loss_a, prob_a, entropy_a= self.train_criterion(pred, y_a, *args)
        loss_b, porb_b, entropy_b = self.train_criterion(pred, y_b, *args)
        return lam * loss_a + (1 - lam) * loss_b, lam * prob_a + (1-lam) * porb_b, lam * entropy_a + (1-lam) * entropy_b
