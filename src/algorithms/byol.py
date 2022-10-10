from os.path import join

from absl import app
from tqdm import tqdm

from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pl_bolts.models.self_supervised.byol.byol_module import BYOL as BYOL_module
from src.algorithms.lightning_benchmark import BenchmarkDataModule, BenchmarkTransforms
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
import torch
from pytorch_lightning import loggers as pl_loggers
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator

class BYOL(AlgorithmSkelton):

    def __init__(self):
        AlgorithmSkelton.__init__(self,'byol')

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):



        parser = ArgumentParser()

        # trainer args
        parser = Trainer.add_argparse_args(parser)

        # model args
        parser = BYOL_module.add_model_specific_args(parser)

        args, unknown = parser.parse_known_args()

        # special byol value
        # args.batch_size = 32
        args.num_classes = dataset_info.num_classes

        # add gpu support
        args.accelerator ='gpu'
        args.devices =1

        # add parameters
        args.log_every_n_steps = 10
        args.max_epochs = 400 if dataset_info.input_size <= 50 else 200
        args.enable_checkpointing=False


        # args.max_epochs = 100
        epochs_finetune = 50
        args.maxpool1 = False
        args.first_conv = False
        args.batch_size = 32

        args.onine_ft = True

        datamodule = BenchmarkDataModule(ds, dataset_info, args.batch_size)
        datamodule.use_unlabeled_data = True
        datamodule.train_transforms = BenchmarkTransforms(dataset_info,train=True,mode='simclr')
        datamodule.val_transforms = BenchmarkTransforms(dataset_info,train=False,mode='simclr')

        model = BYOL_module(**args.__dict__)



        callbacks = []

        csv_logger = pl_loggers.CSVLogger(save_dir="/data/logs/lightning")

        trainer = Trainer.from_argparse_args(args,max_steps=300000, logger=csv_logger, callbacks=callbacks)
        trainer.fit(model, datamodule=datamodule)

        # fine-tune
        finetuner = SSLFineTuner(model, in_features=model.hparams.encoder_out_dim, num_classes=dataset_info.num_classes, epochs=epochs_finetune)
        # update datamodule
        datamodule = BenchmarkDataModule(ds, dataset_info, 32)
        datamodule.use_unlabeled_data = False
        datamodule.train_transforms = BenchmarkTransforms(dataset_info,train=True,mode='finetune')
        datamodule.val_transforms = BenchmarkTransforms(dataset_info,train=False,mode='finetune')

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

    alg = BYOL()
    alg.apply_algorithm()

    alg.report.show()



if __name__ == '__main__':
    app.run(main)