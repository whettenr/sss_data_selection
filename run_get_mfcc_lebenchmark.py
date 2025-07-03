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

from utils import load_checkpoint, save_checkpoint


SAMPLE_RATE=16000
N_MFCC=13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to compute MFCCs
mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
)
mfcc_transform.to(DEVICE)


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

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

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
    


def compute_avg_mfccs(wav_tensor, sample_rate, n_mfcc=13):

    mfcc = mfcc_transform(wav_tensor)

    # Compute deltas
    delta = torchaudio.functional.compute_deltas(mfcc)
    delta2 = torchaudio.functional.compute_deltas(delta)

    # Concatenate [MFCC; delta; delta-delta]
    all_feats = torch.cat([mfcc, delta, delta2], dim=1)

    # Average across time
    avg_feats = all_feats.mean(dim=-1).squeeze().cpu().numpy()
    
    return avg_feats


def get_features(
    dataloader, 
    save_interval_min=5, 
    ckpt_path="feat_checkpoint.pkl",
    feature_function=compute_avg_mfccs,
):
    results, last_saved_batch = load_checkpoint(ckpt_path)
    start_time = time.time()
    i = 0
    for i, batch in enumerate(tqdm(dataloader)):
        if i <= last_saved_batch:
            continue  # Skip already processed batches

        batch = batch.to(DEVICE)
        w, wl = batch.sig
        feats = feature_function(w, SAMPLE_RATE)
        results.update(dict(zip(batch.id, list(feats))))

        # Checkpoint every X minutes
        if (time.time() - start_time) >= save_interval_min * 60:
            save_checkpoint(ckpt_path, results, i)
            start_time = time.time()  # Reset timer

    # Final save
    save_checkpoint(ckpt_path, results, i)



if __name__ == "__main__":
    # parse args from command line
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # data prep
    dl = data_prep(hparams)

    if hparams["feature_function_name"] == "mel":
        feat_fun=compute_avg_mfccs
    else:
        print('error with feature function name')
        exit(1)

    ckpt_name = f"lb_{hparams['feature_function_name']}_{hparams['lebench_subset']}_{hparams['ckpt_path']}"

    # get features
    print('going into get features')
    get_features(
        dataloader=dl, 
        save_interval_min=hparams["save_int"], 
        ckpt_path=ckpt_name,
        feature_function=feat_fun
    )



# import pandas as pd
# df = pd.read_csv("/users/rwhetten/LeBenchmark/sm/train.csv")
# df.duration.describe()
# df = pd.read_csv("/users/rwhetten/LeBenchmark/med/train.csv")
# df.duration.describe()
# df = pd.read_csv("/users/rwhetten/LeBenchmark/med_clean/train.csv")
# df.duration.describe()
# df = pd.read_csv("/users/rwhetten/LeBenchmark/lg/train.csv")
# df.duration.describe()
# df = pd.read_csv("/users/rwhetten/LeBenchmark/xlg/train.csv")
# df.duration.describe()

# df = pd.read_csv("/users/rwhetten/LeBenchmark/sm_new/train.csv")
# df.duration.describe()

# print("=========================")

# import pandas as pd
# df = pd.read_csv("/local_disk/apollon/rwhetten/loquacious_small_train.csv")
# df.duration.describe()

# df = pd.read_csv("/local_disk/apollon/rwhetten/loquacious_medium_train.csv")
# df.duration.describe()

# df = pd.read_csv("/local_disk/apollon/rwhetten/loquacious_large_train.csv")
# df.duration.describe()


# >>> df = pd.read_csv("/users/rwhetten/LeBenchmark/xlg/train.csv")
# >>> df.duration.describe()
# count    2.702357e+06
# mean     1.940130e+01
# std      1.091899e+01
# min      9.375000e-04
# 25%      9.420000e+00
# 50%      1.992300e+01
# 75%      3.000000e+01
# max      3.000000e+01


# >>> df.duration.describe()
# count    9.487877e+06
# mean     9.563535e+00
# std      5.966921e+00
# min      1.000000e+00
# 25%      4.853625e+00
# 50%      7.110000e+00
# 75%      1.462000e+01
# max      3.999994e+01
# Name: duration, dtype: float64