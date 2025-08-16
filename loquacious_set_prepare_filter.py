"""Simple utilities to load the mysterious Loquacious dataset from HuggingFace.
This does not actually prepare the Loquacious dataset. For this, please refer to the dataset_preparation folder.
This only load the prepared dataset to be used in a SpeechBrain recipe.

Authors
-------
 * Titouan Parcollet, 2024
"""

import multiprocessing
import os

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def filter_dataset_by_ids(ds, ids_to_keep):
    """Filters each split of the dataset based on a list of IDs using batch processing."""
    print("STARTING FILTER ...")

    id_set = set(ids_to_keep)
    
    # Use batch=True to process the data in chunks, which is much faster.
    # The filter function receives a dictionary with lists of values for each key.
    # The function must return a list of booleans of the same length as the batch.
    filtered_dataset = ds.filter(lambda examples: [id_value in id_set for id_value in examples['ID']], batched=True)

    return filtered_dataset



def load_datasets(subset, hf_download_folder, hf_caching_dir, ids_to_keep=None):
    """Load and create the HuggingFace dataset for the Loquacious. It must
    have been downloaded manually into hf_download_folder first. This function
    operates in an "offline" mode and will not try to download the dataset.

    Parameters
    ----------
    subset: str
        Name of the subset of interest: one of [large, medium, small, clean]
    hf_download_folder : str
        The path where HF stored the dataset.
    hf_caching_dir : str
        The path where HF will extract (or not if already done) the dataset.
    ids_to_keep : list
        IDs to keep

    Returns
    -------
    Dictionary of HuggingFace dataset. ["train", "dev", "test"]
    """

    try:
        import datasets
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(error)

    # Managing the download dir as HF can be capricious with this.
    logger.info("Loading dataset from: " + str(hf_download_folder))

    nproc = multiprocessing.cpu_count()
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    # datasets.disable_progress_bars()
    hf_data = load_dataset(
        hf_download_folder,
        name=subset,
        num_proc=nproc,
        cache_dir=hf_caching_dir,
    )
    os.environ["HF_DATASETS_OFFLINE"] = "0"

    # Optional: filter based on a list of IDs
    if ids_to_keep is not None:
        hf_data['train'] = filter_dataset_by_ids(hf_data["train"], ids_to_keep)
    return hf_data