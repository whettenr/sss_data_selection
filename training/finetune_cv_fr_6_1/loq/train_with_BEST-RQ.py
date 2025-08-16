#!/usr/bin/env python3
"""
Authors
 * Ryan Whetten 2025
"""

import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        if self.hparams.streaming:
            dynchunktrain_config = self.hparams.dynchunktrain_config_sampler(
                stage
            )
        else:
            dynchunktrain_config = None

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        ### get fbanks and normalize
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)

        feats = self.modules.CNN(feats)
        enc_out = self.modules.enc(
            feats, wav_lens, dynchunktrain_config=dynchunktrain_config
        )
        x = self.modules.back_end_ffn(enc_out)

        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        p_tokens = None
        if stage == sb.Stage.VALID:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            p_tokens = test_searcher(p_ctc, wav_lens)
        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC) given predictions and targets."""

        p_ctc, wav_lens, p_tokens = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            tokens = self.hparams.wav_augment.replicate_labels(tokens)
            tokens_lens = self.hparams.wav_augment.replicate_labels(tokens_lens)

        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage == sb.Stage.VALID:
            # Convert token indices to words
            predicted_words = self.tokenizer(p_tokens, task="decode_from_list")

        elif stage == sb.Stage.TEST:
            predicted_words = [hyp[0].text.split(" ") for hyp in p_tokens]

        if stage != sb.Stage.TRAIN:
            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer(target_words, task="decode_from_list")

            if not isinstance(ids, list):
                ids = ids.tolist()
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)
        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_bestrq, new_lr_bestrq = self.hparams.lr_annealing_bestrq(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )

            if not self.hparams.freeze_bestrq:
                sb.nnet.schedulers.update_learning_rate(
                    self.bestrq_optimizer, new_lr_bestrq
                )

            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_bestrq": old_lr_bestrq,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the bestrq optimizer and model optimizer"
        self.bestrq_optimizer = self.hparams.bestrq_opt_class(
            self.modules.pt_model.parameters()
        )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        # save the optimizers in a dictionary
        # the key will be used in `freeze_optimizers()`
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        if not self.hparams.freeze_bestrq:
            self.optimizers_dict["bestrq_optimizer"] = self.bestrq_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "bestrq_opt", self.bestrq_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


# Define custom data procedure
def dataio_prepare(hparams, tokenizer):
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
    test_data = hf_data_dict["test"].rename_column("ID", "audio_id")

    # We need to get the full list of durations of all samples to enable
    # bucketing from the dynamic batch sampler. We do it that way instead
    # of the usual iterable because the HF dataset ALWAYS open the file
    # when called, which means that the dynamic sampling needs to read the
    # 1.5M audio samples from disk.... using a list instead is much master.
    train_len_list = list(train_data.select_columns("duration")["duration"])
    val_len_list = list(valid_data.select_columns("duration")["duration"])

    train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        train_data,
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        valid_data,
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        test_data,
    )

    # we sort testing/val data to speed up decoding and get better results.
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
    )
    test_data = test_data.filtered_sorted(
        sort_key="duration",
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = read_audio(wav["bytes"])
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("text")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_bos", "tokens_eos", "tokens"],
    )

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

    train_loader_kwargs = {
        "batch_sampler": train_batch_sampler,
        "num_workers": hparams["num_workers"],
    }
    valid_loader_kwargs = {
        "batch_sampler": valid_batch_sampler,
        "num_workers": hparams["num_workers"],
    }

    return (
        train_data,
        valid_data,
        test_data,
        train_loader_kwargs,
        valid_loader_kwargs,
    )


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons_ctc"],
        annotation_train=hparams["train_csv"],
        annotation_read="text",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        # bos_id=hparams["bos_index"],
        # eos_id=hparams["eos_index"],
    )

    # Create the datasets objects as well as tokenization and encoding :-D
    (
        train_data,
        valid_data,
        test_data,
        train_loader_kwargs,
        valid_loader_kwargs,
    ) = dataio_prepare(hparams, tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Load the pretrained model
    if "pretrainer" in hparams.keys() and hparams["hub"] is not None:
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    # Adding objects to trainer.
    asr_brain.tokenizer = tokenizer

    vocab_list = [
        tokenizer.sp.id_to_piece(i) for i in range(tokenizer.sp.vocab_size())
    ]

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"],
        vocab_list=vocab_list,
    )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_loader_kwargs,
        valid_loader_kwargs=valid_loader_kwargs,
    )

    asr_brain.evaluate(
        valid_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    # Test
    asr_brain.evaluate(
        test_data,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
