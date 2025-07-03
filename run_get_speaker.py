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


SAMPLE_RATE=16000
N_MFCC=13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
MODEL.to(DEVICE)
MODEL.eval()

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        print(f"Resumed from batch {state['last_batch_index']}")
        return state["results"], state["last_batch_index"]
    else:
        return {}, -1

def save_checkpoint(path, results, last_batch_index):
    with open(path, "wb") as f:
        pickle.dump({
            "results": results,
            "last_batch_index": last_batch_index
        }, f)
    print(f"[Checkpoint] Saved batch {last_batch_index}, total items: {len(results)}")

# go from HF dict to SB dataloader
def data_prep(data_dict, hparams):
    # We must rename the 'id' column because SpeechBrain sampling use this
    # name for the sampler already, also it's not an id, but an audio_path.
    train_data = hf_data_dict["train"].rename_column("ID", "audio_id")
    # create list of durations for the dynamic batch sampler, for speed
    train_len_list = list(train_data.select_columns("duration")["duration"])
    # create dataset obj
    train_data = sb.dataio.dataset.DynamicItemDataset.from_arrow_dataset(
        train_data,
    )

    datasets = [train_data]

    # create and add pipeline to datasets
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = read_audio(wav["bytes"])
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "audio_id", "sig"],
    )

    # for now just working with one dataset
    dataset = datasets[0]
    dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]

    # create dynamic batch sampler
    train_batch_sampler = DynamicBatchSampler(
        train_data,
        length_func=lambda x: x["duration"],
        lengths_list=train_len_list,
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

    for i, batch in enumerate(tqdm(dataloader)):
        if i <= last_saved_batch:
            continue  # Skip already processed batches

        batch = batch.to(DEVICE)
        w, wl = batch.sig
        feats = feature_function(w)
        results.update(dict(zip(batch.audio_id, list(feats))))

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

    # load dataset 
    hf_data_dict = load_datasets(
        hparams["tls_subset"],
        hparams["hf_hub"],
        hparams["hf_caching_dir"],
    )

    # data prep
    dl = data_prep(hf_data_dict, hparams)

    if hparams["feature_function_name"] == "mel":
        feat_fun=compute_avg_mfccs
    elif hparams["feature_function_name"] == "speaker":
        feat_fun=get_speaker_embeddings
    else:
        print('error with feature function name')
        exit(1)

    ckpt_name = f"{hparams['feature_function_name']}_{hparams['tls_subset']}_{hparams['ckpt_path']}"

    # get features
    get_features(
        dataloader=dl, 
        save_interval_min=hparams["save_int"], 
        ckpt_path=ckpt_name,
        feature_function=feat_fun
    )



