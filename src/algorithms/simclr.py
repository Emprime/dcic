from os.path import join

from absl import app
from tqdm import tqdm

from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR as SimCLR_module
from src.algorithms.lightning_benchmark import BenchmarkDataModule, BenchmarkTransforms
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
import torch
from pytorch_lightning import loggers as pl_loggers

from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_list(name='severities', help='Apply severities to the transforms for scale', default=[0.5])

class SimCLR(AlgorithmSkelton):

    def __init__(self, severity : float):
        AlgorithmSkelton.__init__(self,'simclr') #
        self.severity = severity

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):



        parser = ArgumentParser()

        # model args
        parser = SimCLR_module.add_model_specific_args(parser)
        datamodule = BenchmarkDataModule(ds, dataset_info, 128)


        args, unknown = parser.parse_known_args()

        # special simclr value
        # args.batch_size = 64
        args.num_classes = dataset_info.num_classes
        args.gaussian_blur = True
        args.num_samples = len(datamodule.train_dataloader())
        args.fast_dev_run = False
        # args.dataset=dataset_info.name

        # add gpu support
        args.accelerator ='gpu'
        args.devices =1

        # add parameters
        args.log_every_n_steps = 10
        args.max_epochs = 100
        args.enable_checkpointing=False

        callbacks = []

        epochs_finetune = 50
        args.maxpool1 = False if dataset_info.input_size <= 100 else True
        args.first_conv = False if dataset_info.input_size <= 100 else True


        args.batch_size = 256 if dataset_info.input_size <= 50 else (64 if dataset_info.input_size <= 150 else 32)
        print("specified batch_size", args.batch_size)
        args.dataset = 'cifar10'
        args.optimizer = 'lars'
        args.learning_rate = 1.5
        args.exclude_bn_bias = True
        args.max_epochs = 400 if dataset_info.input_size <= 50 else (200 if dataset_info.input_size <= 150 else 100)
        args.onine_ft = True


        datamodule = BenchmarkDataModule(ds, dataset_info, args.batch_size)
        datamodule.use_unlabeled_data = True
        datamodule.train_transforms = BenchmarkTransforms(dataset_info, train=True, mode='simclr',
                                                          severity=self.severity)
        datamodule.val_transforms = BenchmarkTransforms(dataset_info, train=False, mode='simclr', severity=self.severity)



        model = SimCLR_module(**args.__dict__)


        csv_logger = pl_loggers.CSVLogger(save_dir="/data/logs/lightning")

        trainer = Trainer(
            max_epochs=args.max_epochs,
            max_steps=None if args.max_steps == -1 else args.max_steps,
            gpus=args.gpus,
            num_nodes=args.num_nodes,
            accelerator="ddp" if args.gpus > 1 else None,
            sync_batchnorm=True if args.gpus > 1 else False,
            precision=32 if args.fp32 else 16,
            callbacks=callbacks,
            fast_dev_run=args.fast_dev_run,
            logger=csv_logger
        )
        trainer.fit(model, datamodule=datamodule)

        # fine-tune
        finetuner = SSLFineTuner(model, in_features=model.hparams.hidden_mlp, num_classes=dataset_info.num_classes, epochs=epochs_finetune)
        # update datamodule
        datamodule = BenchmarkDataModule(ds, dataset_info, 32)
        datamodule.use_unlabeled_data = False
        datamodule.train_transforms = BenchmarkTransforms(dataset_info,train=True,mode='finetune', severity=self.severity)
        datamodule.val_transforms = BenchmarkTransforms(dataset_info,train=False,mode='finetune', severity=self.severity)

        # train
        trainer2 = pl.Trainer(accelerator='gpu',devices=1,max_epochs=epochs_finetune, log_every_n_steps=10)
        trainer2.fit(finetuner, datamodule)

        # predict
        predict_dataloader = datamodule.predict_dataloader()

        # prepare model for testing
        device = 'cpu'
        model = model.to(device)
        model.eval()

        all_paths = []
        all_predictions = []
        with torch.no_grad():
            for i, (data, paths) in enumerate(tqdm(predict_dataloader)):
                data = data.to(device)

                feats = finetuner.backbone(data)

                feats = feats.view(feats.size(0), -1)
                logits = finetuner.linear_layer(feats)


                predcitions = torch.softmax(logits, dim=1).cpu().detach().numpy()

                all_paths.extend(paths)
                all_predictions.extend(predcitions)
            # convert to predictions file

        for i, path in enumerate(all_paths):
            split = ds.get(path, 'original_split')  # determine original split before move to unlabeled
            ds.update_image(path, split, [float(temp) for temp in all_predictions[i]])



        return ds

def main(argv):

    report = None
    for severity in FLAGS.severities:

        alg = SimCLR(float(severity))
        if report is None:
            report = alg.report
        else:
            alg.report = report
        alg.apply_algorithm()

    report.show()


if __name__ == '__main__':
    app.run(main)
