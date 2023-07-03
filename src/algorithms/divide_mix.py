from __future__ import print_function
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
import argparse
import numpy as np
import src.algorithms.dividemix_dataloader as dataloader
from sklearn.mixture import GaussianMixture
from absl import app
import numpy as np
from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from absl import app
from absl import flags
# IMPLEMENTATION of DivideMix is based on the official repo https://github.com/LiJunnan1992/DivideMix

FLAGS = flags.FLAGS
flags.DEFINE_string(name='log_directory',
                     help='Log Directory for this methods', default="/data/logs/DivideMix-checkpoint")

class DivideMix(AlgorithmSkelton):
    def __init__(self):
        AlgorithmSkelton.__init__(self, 'divide_mix')


        self.batch_size = 32
        self.lr = 0.002
        self.alpha = 0.5
        self.lambda_u = 0
        self.p_threshold = 0.5
        self.T = 0.5
        self.num_epochs = 80
        self.seed = 123
        self.gpuid = 0
        # self.num_batches = 1000

        torch.cuda.set_device(self.gpuid)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        os.makedirs(FLAGS.log_directory,exist_ok=True)


    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):

        self.exp_id = f"{self.name}-{dataset_info.name}-{num_annos:02d}-{percentage_labeled:0.02f}-{v_fold}"
        self.num_class = dataset_info.num_classes

        log = open(f'{FLAGS.log_directory}/{self.exp_id}.txt', 'w')
        log.flush()

        loader = dataloader.benchmark_loader(ds=ds, dataset_info=dataset_info, batch_size=self.batch_size, num_workers=5
                                             #num_batches=self.num_batches
        )

        print('| Building net')
        net1 = self.create_model()
        net2 = self.create_model()
        cudnn.benchmark = True

        optimizer1 = optim.SGD(net1.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-3)
        optimizer2 = optim.SGD(net2.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-3)

        self.CE = nn.CrossEntropyLoss(reduction='none')
        self.CEloss = nn.CrossEntropyLoss()
        self.conf_penalty = NegEntropy()

        self.best_acc = [0, 0]
        for epoch in range(self.num_epochs + 1):
            lr = self.lr
            if epoch >= 40:
                lr /= 10
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr

            if epoch < 1:  # warm up
                train_loader = loader.run('warmup')
                print('Warmup Net1')
                self.warmup(net1, optimizer1, train_loader)
                train_loader = loader.run('warmup')
                print('\nWarmup Net2')
                self.warmup(net2, optimizer2, train_loader)
            else:
                pred1 = (prob1 > self.p_threshold)  # divide dataset
                pred2 = (prob2 > self.p_threshold)

                print('\n\nTrain Net1')
                labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2,
                                                                        paths=paths2)  # co-divide
                self.train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1
                print('\nTrain Net2')
                labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1,
                                                                        paths=paths1)  # co-divide
                self.train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

            val_loader = loader.run('val')  # validation
            acc1 = self.val(net1, val_loader, 1)
            acc2 = self.val(net2, val_loader, 2)
            log.write('Validation Epoch:%d      Acc1:%.2f  Acc2:%.2f\n' % (epoch, acc1, acc2))
            log.flush()
            print('\n==== net 1 evaluate next epoch training data loss ====')
            self.eval_loader = loader.run('eval_train')  # evaluate training data loss for next epoch
            prob1, paths1 = self.eval_train(epoch, net1)
            print('\n==== net 2 evaluate next epoch training data loss ====')
            self.eval_loader = loader.run('eval_train')
            prob2, paths2 = self.eval_train(epoch, net2)

        all_loader = loader.run('complete')
        net1.load_state_dict(torch.load(f'{FLAGS.log_directory}/{self.exp_id}_net1.pth.tar'))
        net2.load_state_dict(torch.load(f'{FLAGS.log_directory}/{self.exp_id}_net2.pth.tar'))

        self.predict(net1, net2, all_loader, ds)

        # log.write('Test Accuracy:%.2f\n' % (acc))
        # log.flush()

        return ds

    # Training
    def train(self,epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
        net.train()
        net2.eval()  # fix one network and train the other

        unlabeled_train_iter = iter(unlabeled_trainloader)
        num_iter = (len(labeled_trainloader.dataset) // self.batch_size) + 1
        for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
            try:
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, self.num_class).scatter_(1, labels_x.view(-1, 1), 1)
            w_x = w_x.view(-1, 1).type(torch.FloatTensor)

            inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
            inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

            with torch.no_grad():
                # label co-guessing of unlabeled samples
                outputs_u11 = net(inputs_u)
                outputs_u12 = net(inputs_u2)
                outputs_u21 = net2(inputs_u)
                outputs_u22 = net2(inputs_u2)

                pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                            dim=1) + torch.softmax(
                    outputs_u22, dim=1)) / 4
                ptu = pu ** (1 / self.T)  # temparature sharpening

                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                # label refinement of labeled samples
                outputs_x = net(inputs_x)
                outputs_x2 = net(inputs_x2)

                px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                px = w_x * labels_x + (1 - w_x) * px
                ptx = px ** (1 / self.T)  # temparature sharpening

                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

                # mixmatch
            l = np.random.beta(self.alpha, self.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a[:batch_size * 2] + (1 - l) * input_b[:batch_size * 2]
            mixed_target = l * target_a[:batch_size * 2] + (1 - l) * target_b[:batch_size * 2]

            logits = net(mixed_input)

            Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))

            # regularization
            prior = torch.ones(self.num_class) / self.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

            loss = Lx + penalty

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t  Labeled loss: %.4f '
                             % (self.exp_id,epoch, self.num_epochs, batch_idx + 1, num_iter, Lx.item()))
            sys.stdout.flush()

    def warmup(self,net, optimizer, dataloader):
        net.train()
        num_iter = (len(dataloader.dataset) // self.batch_size) + 1
        for batch_idx, (inputs, labels, path) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = self.CEloss(outputs, labels)

            penalty = self.conf_penalty(outputs)
            L = loss + penalty
            L.backward()
            optimizer.step()

            sys.stdout.write('\r')
            sys.stdout.write('|Warm-up: Iter[%3d/%3d]\t CE-loss: %.4f  Conf-Penalty: %.4f'
                             % (batch_idx + 1, num_iter, loss.item(), penalty.item()))
            sys.stdout.flush()

    def val(self,net, val_loader, k):
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        print("\n| Validation\t Net%d  Acc: %.2f%%" % (k, acc))
        if acc > self.best_acc[k - 1]:
            self.best_acc[k - 1] = acc
            print('| Saving Best Net%d ...' % k)
            save_point = f'{FLAGS.log_directory}/{self.exp_id}_net{k}.pth.tar'
            torch.save(net.state_dict(), save_point)
        return acc

    def test(self,net1, net2, test_loader):
        net1.eval()
        net2.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = outputs1 + outputs2
                _, predicted = torch.max(outputs, 1)

                total += targets.size(0)
                correct += predicted.eq(targets).cpu().sum().item()
        acc = 100. * correct / total
        print("\n| Test Acc: %.2f%%\n" % (acc))
        return acc

    def predict(self,net1, net2, all_loader, ds):
        net1.eval()
        net2.eval()
        all_paths = []
        all_predictions = []
        with torch.no_grad():
            for batch_idx, (inputs, paths) in enumerate(all_loader):
                inputs = inputs.cuda()
                outputs1 = net1(inputs)
                outputs2 = net2(inputs)
                outputs = torch.softmax(outputs1 + outputs2, dim=1)# + torch.softmax(outputs2, dim=1)
                _, predicted = torch.max(outputs, 1)
                predcitions = outputs.cpu().detach().numpy()

                # print(paths, predicted, outputs)


                all_paths.extend(paths)
                all_predictions.extend(predcitions)

                # break

        # convert to predictions file

        for i, path in enumerate(all_paths):
            split = ds.get(path, 'original_split')  # determine original split before move to unlabeled
            ds.update_image(path, split, [float(temp) for temp in all_predictions[i]])

        # print(all_paths, all_predictions)
        #         total += targets.size(0)
        #         correct += predicted.eq(targets).cpu().sum().item()
        # acc = 100. * correct / total
        # print("\n| Test Acc: %.2f%%\n" % (acc))

    def eval_train(self,epoch, model):
        model.eval()
        num_samples = len(self.eval_loader.dataset) # self.num_batches * self.batch_size
        losses = torch.zeros(num_samples)
        paths = []
        n = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, path) in enumerate(self.eval_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = self.CE(outputs, targets)
                for b in range(inputs.size(0)):
                    losses[n] = loss[b]
                    paths.append(path[b])
                    n += 1
                sys.stdout.write('\r')
                sys.stdout.write('| Evaluating loss Iter %3d\t' % (batch_idx))
                sys.stdout.flush()

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        losses = losses.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=10, reg_covar=5e-4, tol=1e-2)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmin()]
        return prob, paths

    def create_model(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(2048, self.num_class)
        model = model.cuda()
        return model

class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))

def main(argv):
    """
       Apply only initial annotation
       :return:
    """

    alg = DivideMix()
    alg.apply_algorithm()


if __name__ == '__main__':
    app.run(main)






