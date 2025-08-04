#!/usr/bin/env python3
"""Recipe for pretraining Best-RQ (https://arxiv.org/pdf/2405.04296)

To run this recipe call python train.py BEST-RQ.yaml --find_unused_parameters

Authors
    * Ryan Whetten 2023
"""

import sys
import time
from functools import partial

import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.lobes.models.BESTRQ import brq_mask_collate_fn
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class BestRQBrain(sb.core.Brain):

    def compute_forward(self, batch, stage):
        """Computes forward pass through BestRQ model and returns encoded and
        target embeddings as well as other metrics of interest.
        """

        if self.hparams.streaming:
            dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(
                stage
            )
        else:
            dynchunktrain_config = None


        # get batch and mask
        wavs, wav_lens, mask = batch
        wavs, wav_lens, mask = (
            wavs.to(self.device),
            wav_lens.to(self.device),
            mask.to(self.device),
        )

        ### get fbanks and normalize
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        ### augment data if necessary
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        divis_by = self.hparams.pad_to_divisible_by
        feats = pad_feats(feats, divis_by)

        # get targets from quantizer and stack the frames!
        mask_idx = mask[::4] // 4
        B, T, C = feats.shape
        targets = self.modules.Quantizer(
            feats.view(B, feats.shape[1] // divis_by, -1)[:,mask_idx,:]
        )

        # generate random noise
        noise = torch.normal(
            mean=self.hparams.noise_mean,
            std=self.hparams.noise_std,
            size=(B, mask.shape[0], C),
            device=self.device,
        )
        # replace with random noise
        feats[:, mask, :] = noise

        #### convolutions
        src = self.modules.CNN(feats)

        ##### transformer
        enc_out = self.modules.wrapper(
            src, wav_lens, dynchunktrain_config=dynchunktrain_config
        )  # only use encoder

        ##### linear
        logits = self.modules.linear(enc_out)

        ##### get masked region for loss computation only over these.
        logits = logits[:, mask_idx, :]

        B, T, C = logits.shape
        return logits.view(B * T, C), targets.view(B * T)

    def compute_objectives(self, predictions, batch, stage):
        pred, targets = predictions

        if stage != sb.Stage.TRAIN and sb.utils.distributed.if_main_process():
            predicted_classes = torch.argmax(pred, dim=-1)
            correct_predictions = predicted_classes == targets
            accuracy = correct_predictions.sum().item() / len(
                correct_predictions
            )
            self.acc_metric.append(accuracy)

        return F.cross_entropy(pred, targets)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Called after fit_batch(), updates learning rate and does per-step logging."""

        if should_step:
            self.hparams.noam_annealing(self.optimizer)

        # Perform step-wise logging
        if (
            hasattr(self.hparams, "log_interval")
            and self.optimizer_step % self.hparams.log_interval == 0
        ):

            # Create a dictionary and fill it with everything we
            # want to log such as contrastive loss, diversity loss,
            # learning rate etc.
            log_dct = {}

            current_lr = self.optimizer.param_groups[0]["lr"]
            log_dct["steps"] = self.optimizer_step
            log_dct["lr"] = current_lr
            log_dct["avg_loss"] = self.avg_train_loss

            if hasattr(self, "time_last_log"):
                run_time_since_last_log = time.time() - self.time_last_log
                log_dct["run_time"] = run_time_since_last_log
            self.time_last_log = time.time()

            if sb.utils.distributed.if_main_process():
                self.hparams.train_steps_logger.log_stats(
                    stats_meta=log_dct,
                )

        # Perform step-wise checkpoint saving
        if (self.optimizer_step in self.hparams.save_interval):
            self.checkpointer.checkpoints_dir = self.checkpointer.checkpoints_dir.parent / 'steps'
            self.checkpointer.save_checkpoint(
                end_of_epoch=False,
                name='step_' + str(self.optimizer_step)
            )
            # move checkpoint back so that it will continue to train as normal
            self.checkpointer.checkpoints_dir = self.checkpointer.checkpoints_dir.parent / 'save'

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []

    def on_stage_end(self, stage, stage_loss, epoch=None):

        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            if self.acc_metric:

                stage_stats["accuracy"] = sum(self.acc_metric) / len(
                    self.acc_metric
                )

            self.hparams.train_stage_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": self.optimizer_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                end_of_epoch=True,
                num_to_keep=5,
                meta={"valid_loss": stage_loss},
            )


def pad_feats(feats, divis_by):
    """BEST-RQ quantizer stackes frames together. Hence, we need to pad the
    incoming features such that the time dimension is divisible by divis_by.

    Arguments
    ---------
    feats: torch.Tensor
        The feature tensor.
    divis_by: int
        The stacking factor. The time dimension of feats will become divisible
        by this value.

    Returns
    -------
    Padded features
    """

    B, T, C = feats.shape

    #### pad features to enable a reduction by pad_to_divisible_by for the
    # quantiser of BEST-RQ
    current_dim_size = T
    dim_to_pad = 1  # Pad along the second dimension (i.e. time)

    # Calculate the amount of padding needed to make the tensor divisible
    # by divis_by
    current_dim_size = feats.shape[dim_to_pad]
    # Ensure positive padding
    padding_needed = (divis_by - (current_dim_size % divis_by)) % divis_by

    # Define the padding
    # Initialize padding for all dimensions, have a look at the documentation of
    # torch.nn.functional.pad because the padding argument is quite special.
    padding = [0, 0, 0, 0, 0, 0]
    padding[dim_to_pad * 2] = (
        padding_needed  # Set padding for the chosen dimension
    )

    # add in padding to features and mask
    return torch.nn.functional.pad(feats, padding)


def dataio_prepare(hparams):
    
    from loquacious_set_prepare_filter import load_datasets

    if hparams['filter']:
        import pandas as pd
        df = pd.read_csv(hparams['train_csv'])
        
        hf_data_dict = load_datasets(
            hparams["tls_subset"],
            hparams["hf_hub"],
            hparams["hf_caching_dir"],
            ids_to_keep=list(df.ID),
        )
        
    else:
        hf_data_dict = load_datasets(
            hparams["tls_subset"],
            hparams["hf_hub"],
            hparams["hf_caching_dir"],
        )


    # We must rename the 'id' column because SpeechBrain sampling use this
    # name for the sampler already, also it's not an id, but an audio_path.
    train_data = hf_data_dict["train"].rename_column("ID", "audio_id")
    valid_data = hf_data_dict["dev"].rename_column("ID", "audio_id")
    # test_data = hf_data_dict["test"].rename_column("ID", "audio_id")

    # We need to get the full list of durations of all samples to enable
    # bucketing from the dynamic batch sampler. We do it that way instead
    # of the usual iterable because the HF dataset ALWAYS open the file
    # when called, which means that the dynamic sampling needs to read the
    # 1.5M audio samples from disk.... using a list instead is much faster.
    train_len_list = list(train_data.select_columns("duration")["duration"])
    val_len_list = list(valid_data.select_columns("duration")["duration"])

    train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        train_data,
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        valid_data,
    )

    # test_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
    #     test_data,
    # )


    datasets = [train_data, valid_data]
    # datasets = [train_data, valid_data, test_data]

    def get_output_lengths(input_lengths):
        """Function to get the output length of the feature extractor this is
        necessary to compute the masks of BestRQ.
        """
        sr = hparams["sample_rate"]
        hop_length = hparams["hop_length"]

        return (input_lengths // (sr * hop_length / 1000) + 1).to(torch.long)

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav["bytes"])
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    # 5. we instantiate the needed samplers with dynamic batching
    dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
    dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

    train_batch_sampler = DynamicBatchSampler(
        train_data,
        length_func=lambda x: x["duration"],
        lengths_list=train_len_list,
        **dynamic_hparams_train,
    )

    valid_batch_sampler = DynamicBatchSampler(
        valid_data,
        length_func=lambda x: x["duration"],
        lengths_list=val_len_list,
        **dynamic_hparams_valid,
    )



    # We define the custom collation function that is necessary for best-rq to
    # generate masks.
    brq_mask_collate_fn_partial = partial(
        brq_mask_collate_fn,
        get_out_len_fn=get_output_lengths,
        mask_prob=hparams["mask_prob"],
        mask_length=hparams["mask_length"],
        n_mels=hparams["n_mels"],
    )

    train_loader_kwargs = {
        "batch_sampler": train_batch_sampler,
        "collate_fn": brq_mask_collate_fn_partial,
        "num_workers": hparams["num_workers"],
    }

    valid_loader_kwargs = {"batch_sampler": valid_batch_sampler}

    return (
        train_data,
        valid_data,
        train_loader_kwargs,
        valid_loader_kwargs,
    )


def main():
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams.update(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects
    (
        train_data,
        valid_data,
        train_loader_kwargs,
        valid_loader_kwargs,
    ) = dataio_prepare(hparams)

    brain = BestRQBrain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # with torch.autograd.detect_anomaly():
    brain.fit(
        brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=valid_loader_kwargs,
        progressbar=True,
    )


if __name__ == "__main__":
    main()
