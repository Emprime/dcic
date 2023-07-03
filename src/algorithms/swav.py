from os.path import join

from absl import app
from tqdm import tqdm

from src.algorithms.common.algorithm_skeleton import AlgorithmSkelton
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pl_bolts.models.self_supervised.swav.swav_module import SwAV as SWAV_module
# from src.algorithms.swav_module import SwAV as SWAV_module
from src.algorithms.lightning_benchmark import BenchmarkDataModule, BenchmarkTransforms
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
import torch
from pytorch_lightning import loggers as pl_loggers


class SWAV(AlgorithmSkelton):

    def __init__(self):
        AlgorithmSkelton.__init__(self,'swav')

    def run(self, ds, oracle, dataset_info, v_fold,num_annos,percentage_labeled):



        parser = ArgumentParser()

        # model args
        parser = SWAV_module.add_model_specific_args(parser)



        args, unknown = parser.parse_known_args()

        # special swav value
        args.batch_size = 16 if dataset_info.input_size <= 112 else 32

        datamodule = BenchmarkDataModule(ds, dataset_info, args.batch_size)
        args.num_classes = dataset_info.num_classes
        args.num_samples = len(datamodule.train_dataloader())
        args.fast_dev_run = False
        args.dataset=dataset_info.name
        # arguments also mnaula added in `lighning_benchmark.py`
        args.size_crops = [192, 96] if dataset_info.input_size <= 112 else [192,64]
        args.nmb_crops = [2, 1] if dataset_info.input_size <= 112 else [1,2]
        args.min_scale_crops = [0.5, 0.3]
        args.max_scale_crops = [1.0, 0.5]

        # add gpu support
        args.accelerator ='gpu'
        args.devices =1

        # add parameters
        args.log_every_n_steps = 10
        args.max_epochs = 100
        args.enable_checkpointing=False

        args.maxpool1 = False
        args.first_conv = False
        args.dataset = 'cifar10'
        args.optimizer = 'lars'
        args.learning_rate = 1.5
        args.onine_ft = True

        datamodule = BenchmarkDataModule(ds, dataset_info, args.batch_size)
        datamodule.use_unlabeled_data = True
        datamodule.train_transforms = BenchmarkTransforms(dataset_info, train=True, mode='swav')
        datamodule.val_transforms = BenchmarkTransforms(dataset_info, train=False, mode='swav')


        model = SWAV_module(**args.__dict__)

        # print(model.hparams)
        # print(model)

        callbacks = []
        # online_evaluator = SSLOnlineEvaluator(
        #     z_dim=model.hparams.hidden_mlp,
        #     num_classes=args.num_classes,
        #     dataset=args.dataset,
        # )

        # callbacks.append(online_evaluator)

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
        epochs = 50
        finetuner = SSLFineTuner(model, in_features=model.hparams.hidden_mlp, num_classes=dataset_info.num_classes, epochs=epochs)
        # update datamodule
        datamodule = BenchmarkDataModule(ds, dataset_info, 32)
        datamodule.use_unlabeled_data = False
        datamodule.train_transforms = BenchmarkTransforms(dataset_info,train=True,mode='finetune')
        datamodule.val_transforms = BenchmarkTransforms(dataset_info,train=False,mode='finetune')

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

    alg = SWAV()
    alg.apply_algorithm()
    alg.report.show()


if __name__ == '__main__':
    app.run(main)