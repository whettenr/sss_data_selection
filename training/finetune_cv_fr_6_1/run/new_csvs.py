# module load pytorch-gpu/py3/2.1.1
# conda activate ft-sb
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_voice_prepare import prepare_common_voice

data_folder = "/lustre/fsmisc/dataset/CommonVoice/cv-corpus-6.1-2020-12-11/fr"

hparams = {
    "data_folder": data_folder,
    "save_folder": "/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/finetune_cv_fr_6_1/new_csvs",
    "train_tsv_file": f"{data_folder}/train.tsv",
    "dev_tsv_file": f"{data_folder}/dev.tsv",
    "test_tsv_file": f"{data_folder}/test.tsv",
    "accented_letters": True,
    "language": "fr",
    "skip_prep": False,
}

prepare_common_voice(**hparams)


