import os
import sys
import time
import pickle
import torch
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
from loquacious_set_prepare import load_datasets
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.sampler import DynamicBatchSampler

import torchaudio
import torchaudio.transforms as T
from transformers.models.seamless_m4t.feature_extraction_seamless_m4t import SeamlessM4TFeatureExtractor

from torch.cuda.amp import autocast
import torch.profiler
import numpy as np
from speechbrain.utils.checkpoints import Checkpointer
import yaml

from speechbrain.utils.logger import get_logger
logger = get_logger(__name__)


SAMPLE_RATE=16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# def get_last_saved_batch(log_path):
#     if os.path.exists(log_path):
#         with open(log_path, "r") as f:
#             return int(f.read().strip())
#     return -1

# def set_last_saved_batch(log_path, batch_index):
#     with open(log_path, "w") as f:
#         f.write(str(batch_index))

def save_batch_features(audio_ids, feats, batch_index, save_dir):
    save_path = os.path.join(save_dir, f"features_batch_{batch_index:05d}.pkl")
    data = dict(zip(audio_ids, feats))
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[Saved] {save_path} with {len(data)} items.")

def data_prep(hparams):

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
    )

    # We remove longer and shorter files from the train.
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    
    datasets = [train_data]
    

    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav,  start, stop):
        sig = sb.dataio.dataio.read_audio({
            "file": wav,
            "start": int(start),
            "stop": int(stop),
        })
        np_wav = np.array(sig)
        features = hparams["feature_extractor"](np_wav, sampling_rate=16000)["input_features"][0]
        return torch.Tensor(features)

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig"])

    dataset = datasets[0]
    dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]

    # create dynamic batch sampler
    train_batch_sampler = DynamicBatchSampler(
        train_data,
        length_func=lambda x: x["duration"],
        **dynamic_hparams_train,
    )

    train_loader_kwargs = {
        "batch_sampler": train_batch_sampler,
        "num_workers": hparams["num_workers"],
    }

    # create dataloader
    dataloader = sb.dataio.dataloader.make_dataloader(
        dataset, **train_loader_kwargs
    )
    print('len dl: ', len(dataloader))

    return dataloader
    

def get_features(
    dataloader,
    save_interval_min=5,
    save_dir="feature_checkpoints",
    log_file="last_batch.txt",
    w2v_bert=None,
    attention_pool=None,
    ckpt_path="feature_checkpoints/ckpts",
    profiler=False,
):
    os.makedirs(save_dir, exist_ok=True)

    if w2v_bert is None or attention_pool is None:
        print("issue with model")
        exit(1)

    # last_saved_batch = get_last_saved_batch(os.path.join(save_dir, log_file))
    start_time = time.time()

    audio_id_accum = []
    feat_accum = []

    step = Step(0)
    checkpointer = Checkpointer(ckpt_path, {"dataloader": dataloader, "step": step})
    _ = checkpointer.recover_if_possible()
    print(f"going from step: {step.step} to {step.end}")
    print(ckpt_path)

    if not profiler:
        with tqdm(
            dataloader,
            initial=step.step,
            dynamic_ncols=True,
        ) as t:
            for batch in t:
                step()
                w, wl = batch.sig
                w = w.to(DEVICE)
                with torch.no_grad():
                    with autocast(dtype=torch.float16):
                        feats = w2v_bert(w)
                        feats = attention_pool(feats)

                audio_id_accum.extend(batch.id)
                feat_accum.extend(feats.cpu().numpy())

                if (time.time() - start_time) >= save_interval_min * 60:
                    save_batch_features(audio_id_accum, feat_accum, step.step - 1, save_dir)
                    # set_last_saved_batch(os.path.join(save_dir, log_file), i)
                    _ = checkpointer.save_checkpoint(end_of_epoch = False)
                    audio_id_accum.clear()
                    feat_accum.clear()
                    start_time = time.time()

                if (step.end > 0) and  step.end == (step.step - 1):
                    save_batch_features(audio_id_accum, feat_accum, step.step - 1, save_dir)
                    _ = checkpointer.save_checkpoint(end_of_epoch = False)
                    audio_id_accum.clear()
                    feat_accum.clear()
                    break

        # Final save if anything remains
        if audio_id_accum:
            save_batch_features(audio_id_accum, feat_accum, step.step - 1, save_dir)
            # set_last_saved_batch(os.path.join(save_dir, log_file), i)

    else:

        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=20, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/no_d_update_batch_1200'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            for i, batch in enumerate(tqdm(dataloader)):

                prof.step()
                # UNCOMMENT
                # if i <= last_saved_batch:
                #     continue  # Skip already processed batches
                batch = batch.to(DEVICE)
                w, wl = batch.sig
                # w, wl = batch.sig
                # w = w.to(DEVICE)
                with torch.no_grad():
                    with autocast(dtype=torch.float16):
                        feats = w2v_bert(w)
                        feats = attention_pool(feats)
                
                audio_id_accum.extend(batch.audio_id)
                feat_accum.extend(feats.cpu().numpy())

                if (time.time() - start_time) >= save_interval_min * 60:
                    save_batch_features(audio_id_accum, feat_accum, i, save_dir)
                    set_last_saved_batch(os.path.join(save_dir, log_file), i)
                    audio_id_accum.clear()
                    feat_accum.clear()
                    start_time = time.time()
                
                if i == 1 + 2 + 20:
                    break

        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

@sb.utils.checkpoints.register_checkpoint_hooks
class Step:
    def __init__(self,step=0,end=0):
        self.step = step
        self.end = end

    def __call__(self):
        self.step = self.step + 1

    def __str__(self):
        return f"step count: {self.step}"
        
    @sb.utils.checkpoints.mark_as_saver
    def _save(self, path):
        save_dict = {
            "step": self.step,
            "end": self.end,
        }
        with open(path, "w", encoding="utf-8") as w:
            w.write(yaml.dump(save_dict))

    @sb.utils.checkpoints.mark_as_loader
    def _recover(self, path, end_of_epoch):
        del end_of_epoch
        with open(path, encoding="utf-8") as f:
            save_dict = yaml.safe_load(f)
        self.step = save_dict["step"]
        self.end = save_dict["end"]


if __name__ == "__main__":
    # parse args from command line
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # data prep
    dl = data_prep(hparams)

    if hparams["feature_function_name"] != "sense":
        print('error with feature function name')
        print('use sense')
        exit(1)


    if "pretrainer" in hparams.keys():
        print("loading weights...")
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    w2v_bert = hparams["wav2vec2"]
    attention_pool = hparams["model"][0]
    print(f"using {DEVICE}")
    w2v_bert.to(DEVICE)
    w2v_bert.eval()
    attention_pool.to(DEVICE)
    attention_pool.eval()

    save_dir = f"lb_{hparams['feature_function_name']}_{hparams['lebench_subset']}_{hparams['save_dir']}"
    ckpt_path = f"lb_{hparams['feature_function_name']}_{hparams['lebench_subset']}_{hparams['save_dir']}/{hparams['ckpt_path']}"

    # get features
    get_features(
        dataloader=dl, 
        save_interval_min=hparams["save_int"], 
        save_dir=save_dir,
        log_file="last_batch.txt",
        w2v_bert=w2v_bert,
        attention_pool=attention_pool,
        ckpt_path=ckpt_path,
        profiler=hparams["profiler"]
    )



