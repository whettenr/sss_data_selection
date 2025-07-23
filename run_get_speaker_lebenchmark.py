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
from pyannote.audio import Model

import torchaudio
import torchaudio.transforms as T

from utils import load_checkpoint, save_checkpoint


SAMPLE_RATE=16000
N_MFCC=13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
MODEL.to(DEVICE)
MODEL.eval()


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
    


def get_speaker_embeddings(wavs):
    with torch.no_grad():
        embeddings = MODEL(wavs.unsqueeze(1))
    return embeddings.cpu().numpy()


def get_features(
    dataloader, 
    save_interval_min=5, 
    ckpt_path="feat_checkpoint.pkl",
    feature_function=get_speaker_embeddings,
):
    results, last_saved_batch = load_checkpoint(ckpt_path)
    start_time = time.time()
    i = 0
    for i, batch in enumerate(tqdm(dataloader)):
        if i <= last_saved_batch:
            continue  # Skip already processed batches

        batch = batch.to(DEVICE)
        w, wl = batch.sig
        feats = feature_function(w)
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
    elif hparams["feature_function_name"] == "speaker":
        feat_fun=get_speaker_embeddings
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