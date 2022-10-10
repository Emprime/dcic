from __future__ import print_function
import src.algorithms.elr_files.data_loaders as dl
from src.algorithms.elr_files.trainer import Trainer
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.models as models
import random
import os
from tqdm import tqdm
import numpy as np
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from absl import app
from absl import flags
# IMPLEMENTATION of ELR Plus is based on the official repo https://github.com/shengliu66/ELR with the clothing1m Dataset

FLAGS = flags.FLAGS
flags.DEFINE_string(name='log_directory',
                     help='Log Directory for this methods', default="/data/logs/elr-checkpoint")

class ELR_Plus(AlgorithmSkelton):
    def __init__(self):
        AlgorithmSkelton.__init__(self, 'elr')

        self.batch_size = 64

        # optimizer
        self.lr = 0.002
        self.momentum = 0.9
        self.weight_decay = 1e-3

        # loss
        self.beta = 0.7
        self._lambda = 3

        # lr_scheduler
        self.milestones = [7]
        self.gamma = 0.1

        # data loader
        self.num_workers = 8
        self.pin_memory = True
        self.shuffle = True

        # other parameters
        self.mixup_alpha = 1
        self.coef_step = 0
        self.ema_alpha =0.9999
        self.ema_update = False
        self.ema_step = -1

        # trainer
        self.epochs = 15
        self.warmup = 0
        self.save_dir = FLAGS.log_directory
        self.verbosity = 2
        self.early_stop = 2000
        self.begin = 0
        self.asym = False
        self.percent = 0.8
        self.monitor = "max val_my_metric"

        os.makedirs(self.save_dir, exist_ok=True)


        self.seed = 123
        self.gpuid = 0
        # self.num_batches = 1000

        torch.cuda.set_device(self.gpuid)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)



    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):

        self.num_class = dataset_info.num_classes


        # setup data_loader instances
        data_loader1 = dl.AdvancedDataLoader(
            ds, dataset_info,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_split=0,
            training=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        data_loader2 = dl.AdvancedDataLoader(
            ds, dataset_info,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            validation_split=0,
            training=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

        valid_data_loader = data_loader1.split_validation()

        # build model architecture
        model1 = self.resnet50(self.num_class)
        model_ema1 = self.resnet50(self.num_class)
        model_ema1_copy = self.resnet50(self.num_class)
        model2 = self.resnet50(self.num_class)
        model_ema2 = self.resnet50(self.num_class)
        model_ema2_copy = self.resnet50(self.num_class)

        # get function handles of loss and metrics
        device_id = list(range(min(torch.cuda.device_count(), 1)))

        if hasattr(data_loader1.dataset, 'num_raw_example') and hasattr(data_loader2.dataset, 'num_raw_example'):
            num_examp1 = data_loader1.dataset.num_raw_example
            num_examp2 = data_loader2.dataset.num_raw_example
        else:
            num_examp1 = len(data_loader1.dataset)
            num_examp2 = len(data_loader2.dataset)

        train_loss1 = elr_plus_loss(num_examp=num_examp1,
                                             num_classes=self.num_class,
                                             device='cuda:' + str(device_id[0]),
                                             beta=self.beta, coef_step=self.coef_step, _lambda= self._lambda)
        train_loss2 = elr_plus_loss(num_examp=num_examp2,
                                             num_classes=self.num_class,
                                             device='cuda:' + str(device_id[-1]),
                                             beta=self.beta, coef_step=self.coef_step, _lambda= self._lambda)

        val_loss = self.cross_entropy
        metrics = [self.my_metric, self.my_metric2]

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params1 = filter(lambda p: p.requires_grad, model1.parameters())
        trainable_params2 = filter(lambda p: p.requires_grad, model2.parameters())

        optimizer1 = torch.optim.SGD(params=trainable_params1, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        optimizer2 = torch.optim.SGD(params=trainable_params2, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1,milestones=self.milestones, gamma=self.gamma)
        lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2,milestones=self.milestones, gamma=self.gamma)

        trainer = Trainer(model1, model2, model_ema1, model_ema2, train_loss1, train_loss2,
                          metrics,
                          optimizer1, optimizer2,
                          data_loader1=data_loader1,
                          data_loader2=data_loader2,
                          valid_data_loader=valid_data_loader,
                          test_data_loader=None,
                          lr_scheduler1=lr_scheduler1,
                          lr_scheduler2=lr_scheduler2,
                          val_criterion=val_loss,
                          model_ema1_copy=model_ema1_copy,
                          model_ema2_copy=model_ema2_copy,
                          epochs=self.epochs, save_period=1, monitor=self.monitor,
                          early_stop=self.early_stop, save_dir=self.save_dir, num_classes=self.num_class,
                          ema_step=self.ema_step, mixup_alpha=self.mixup_alpha, ema_alpha=self.ema_alpha,
                          ema_update=self.ema_update)
        trainer.train()

        data_loader = dl.AdvancedDataLoader(
            ds, dataset_info,
            batch_size=self.batch_size,
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=self.num_workers
        ).split_validation()

        # build model architecture
        model = self.resnet50(self.num_class)

        # get function handles of loss and metrics
        loss_fn = self.cross_entropy
        metric_fns = [self.my_metric, self.my_metric2]

        load_path = self.save_dir + '/model_best.pth'
        print('Loading checkpoint: {} ...'.format(load_path))
        checkpoint = torch.load(load_path, map_location='cpu')
        state_dict = checkpoint['state_dict1']
        # if config['n_gpu'] > 1:
        #     model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)

        # prepare model for testing
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))
        all_paths = []
        all_predictions = []
        with torch.no_grad():
            for i, (data, _, paths, _) in enumerate(tqdm(data_loader)):
                data = data.to(device) #, target.to(device)
                output = model(data)

                #
                # save sample images, or do something with output here
                #
                predcitions = torch.softmax(output, dim=1).cpu().detach().numpy()
                # print(output,paths)

                all_paths.extend(paths)
                all_predictions.extend(predcitions)

            # break

            # convert to predictions file

        for i, path in enumerate(all_paths):
            split = ds.get(path, 'original_split')  # determine original split before move to unlabeled
            ds.update_image(path, split, [float(temp) for temp in all_predictions[i]])

        return ds

    def resnet50(self,num_classes):
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        return model_ft

    def cross_entropy(self,output, target, M=3):
        return F.cross_entropy(output, target)

    def my_metric(self,output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)

    def my_metric2(self,output, target, k=3):
        try:
            with torch.no_grad():
                pred = torch.topk(output, k, dim=1)[1]
                assert pred.shape[0] == len(target)
                correct = 0
                for i in range(k):
                    correct += torch.sum(pred[:, i] == target).item()
            return correct / len(target)
        except RuntimeError:
            return -1


class elr_plus_loss(nn.Module):
        def __init__(self, num_examp, device,coef_step, _lambda, num_classes=10, beta=0.3):
            super(elr_plus_loss, self).__init__()
            self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
            self.q = 0
            self.beta = beta
            self.num_classes = num_classes
            self.coef_step=coef_step
            self._lambda=_lambda

        def forward(self, iteration, output, y_labeled):
            y_pred = F.softmax(output, dim=1)

            y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)

            if self.num_classes == 100:
                y_labeled = y_labeled * self.q
                y_labeled = y_labeled / (y_labeled).sum(dim=1, keepdim=True)

            ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim=-1))
            reg = ((1 - (self.q * y_pred).sum(dim=1)).log()).mean()
            final_loss = ce_loss + self.sigmoid_rampup(iteration, self.coef_step) * (
                        self._lambda* reg)

            return final_loss, y_pred.cpu().detach()

        def update_hist(self, epoch, out, index=None, mix_index=..., mixup_l=1):
            y_pred_ = F.softmax(out, dim=1)
            self.pred_hist[index] = self.beta * self.pred_hist[index] + (1 - self.beta) * y_pred_ / (y_pred_).sum(dim=1,
                                                                                                                  keepdim=True)
            self.q = mixup_l * self.pred_hist[index] + (1 - mixup_l) * self.pred_hist[index][mix_index]

        def sigmoid_rampup(self,current, rampup_length):
            """Exponential rampup from  2"""
            if rampup_length == 0:
                return 1.0
            else:
                current = np.clip(current, 0.0, rampup_length)
                phase = 1.0 - current / rampup_length
                return float(np.exp(-5.0 * phase * phase))


def main(argv):
    alg = ELR_Plus()
    alg.apply_algorithm()


if __name__ == '__main__':
    app.run(main)

