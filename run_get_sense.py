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



SAMPLE_RATE=16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        feature_size = 80
        sampling_rate = 16000
        num_mel_bins = 80
        padding_value = 0.0
        stride = 2
        
        feature_extractor = SeamlessM4TFeatureExtractor(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            num_mel_bins=num_mel_bins,
            padding_value=padding_value,
            stride=stride
        )
        
        sig = read_audio(wav["bytes"])
        # np_wav = np.array(sig)
        features = feature_extractor(sig,sampling_rate=16000)["input_features"][0]
        return torch.Tensor(features)

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
    


def get_features(
    dataloader, 
    save_interval_min=5, 
    ckpt_path="feat_checkpoint.pkl",
    w2v_bert=None,
    attention_pool=None,
):
    if w2v_bert is None or attention_pool is None:
        print("issue with model")
        exit(0)
    results, last_saved_batch = load_checkpoint(ckpt_path)
    start_time = time.time()

    for i, batch in enumerate(tqdm(dataloader)):
        if i <= last_saved_batch:
            continue  # Skip already processed batches

        batch = batch.to(DEVICE)
        w, wl = batch.sig
        with torch.no_grad():
            feats = w2v_bert(w)
            feats = attention_pool(feats)
        
        results.update(dict(zip(batch.audio_id, list(feats.cpu().numpy()))))

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


    ckpt_name = f"{hparams['feature_function_name']}_{hparams['tls_subset']}_{hparams['ckpt_path']}"
    print(DEVICE)
    exit(0)
    # get features
    get_features(
        dataloader=dl, 
        save_interval_min=hparams["save_int"], 
        ckpt_path=ckpt_name,
        w2v_bert=w2v_bert,
        attention_pool=attention_pool,
    )



