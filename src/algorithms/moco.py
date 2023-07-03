from os.path import join

from absl import app
from tqdm import tqdm

from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
# from pl_bolts.models.self_supervised.moco.moco2_module import Moco_v2
from src.algorithms.lightning_benchmark import BenchmarkDataModule, BenchmarkTransforms
from src.algorithms.moco_module import Moco_v2
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
import torch
from pytorch_lightning import loggers as pl_loggers
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_list(name='severities', help='Apply severities to the transforms for scale', default=[0.5])


class MOCO(AlgorithmSkelton):

    def __init__(self, severity : float):
        AlgorithmSkelton.__init__(self,f'moco')
        self.severity = severity

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):



        parser = ArgumentParser()

        # trainer args
        parser = Trainer.add_argparse_args(parser)

        # model args
        parser = Moco_v2.add_model_specific_args(parser)

        args, unknown = parser.parse_known_args()

        # special moco value
        args.batch_size = 32 #256
        args.num_negatives = args.batch_size * 10

        # add gpu support
        args.accelerator ='gpu'
        args.devices =1

        # add parameters
        args.log_every_n_steps = 10
        args.max_epochs = 400
        args.enable_checkpointing=False
        args.dataset = 'cifar10'
        args.num_classes = dataset_info.num_classes


        datamodule = BenchmarkDataModule(ds, dataset_info, args.batch_size)
        datamodule.use_unlabeled_data = True
        datamodule.train_transforms = BenchmarkTransforms(dataset_info,train=True,mode='simclr', severity=self.severity)
        datamodule.val_transforms = BenchmarkTransforms(dataset_info,train=False,mode='simclr', severity=self.severity)

        # check for changed batch_size
        if datamodule.batch_size != args.batch_size:
            args.num_negatives = datamodule.batch_size * 10

        model = Moco_v2(**args.__dict__)

        callbacks = []

        csv_logger = pl_loggers.CSVLogger(save_dir="/data/logs/lightning")

        trainer = Trainer.from_argparse_args(args, logger=csv_logger, callbacks=callbacks)
        trainer.fit(model, datamodule=datamodule)

        # fine-tune
        epochs = 50
        finetuner = SSLFineTuner(model, in_features=model.hparams.emb_dim, num_classes=dataset_info.num_classes, epochs=epochs)
        # update datamodule
        datamodule = BenchmarkDataModule(ds, dataset_info, 32, drop_last=True)
        datamodule.use_unlabeled_data = False
        datamodule.train_transforms = BenchmarkTransforms(dataset_info,train=True,mode='finetune', severity=self.severity)
        datamodule.val_transforms = BenchmarkTransforms(dataset_info,train=False,mode='finetune', severity=self.severity)

        # train
        trainer2 = pl.Trainer(accelerator='gpu',devices=1,max_epochs=epochs, log_every_n_steps=10)
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

        alg = MOCO(float(severity))
        if report is None:
            report = alg.report
        else:
            alg.report = report
        alg.apply_algorithm()

    report.show()



if __name__ == '__main__':
    app.run(main)