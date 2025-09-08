/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2010/20100419-0900-PLENARY-18_fr_34.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/MPF/output_waves/Nacer2_ANON_071017_s_310_spk_Mouna.wav


/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Mass/output_waves/B19___12_Hebreux_____FRNTLSN2DA_verse_6.wav
/corpus/LeBenchmark/MaSS/MaSS/output_waves/B19___12_Hebreux_____FRNTLSN2DA_verse_6.wav


/corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_DIA_1221_166.854_170.146.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20031203_1155_1240_RFI_ELDA_s_62.wav


scp /corpus/LeBenchmark/mpf_flowbert/MPF/output_waves/Nawal6_100419_ANON_s_68_spk_Esma.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/MPF/output_waves/Nawal6_100419_ANON_s_68_spk_Esma.wav

scp -r /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns


scp -r /corpus/LeBenchmark/mpf_flowbert/MPF/output_waves/ \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/MPF/output_waves/

scp -r /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2010/ \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/wav/2010/20101019-0900-PLENARY-13_fr_19.wav

scp -r /corpus/LeBenchmark/voxpopuli_unlabelled/wav/ \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/

scp -r /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/ \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/



./prep_csvs.sh \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_mfcc_0.5.csv \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_mfcc_0.5_fix.csv


./prep_csvs.sh \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_random_0.5.csv \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_random_0.5_fix.csv

./prep_csvs.sh \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_sense_0.5.csv \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_sense_0.5_fix.csv


./prep_csvs.sh \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_speaker_0.5.csv \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_speaker_0.5_fix.csv

./prep_csvs.sh \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_train.csv \
    /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_train_fix.csv


./prep_csvs.sh /users/rwhetten/LeBenchmark/lg/train.csv /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_train.csv
./prep_csvs.sh /users/rwhetten/LeBenchmark/xlg/mls_french-dev.csv /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_mls_french-dev.csv


./prep_csvs.sh /users/rwhetten/LeBenchmark/lg/train.csv /local_disk/apollon/rwhetten/sss_data_selection/training/csvs/lb_lg/jz_train.csv










import pandas as pd
import os
from tqdm import tqdm


# Load CSV
csv_path="/lustre/fswork/projects/rech/nkp/uaj64gk/dataselection/csvs/lb_lg/jz_train_fix.csv"
df = pd.read_csv(csv_path)  # replace with your actual CSV file name

file_paths = df["wav"].tolist()

missing_files = []

# Use tqdm to show progress
for path in tqdm(file_paths, desc="Checking files"):
    if not os.path.exists(path):
        missing_files.append(path)

# Save missing files to a text file
with open("missing_files.txt", "w") as f:
    f.write(f"Total files: {len(file_paths)}\n")
    f.write(f"Missing files: {len(missing_files)}\n\n")
    for path in missing_files:
        f.write(path + "\n")

print(f"Check complete. {len(missing_files)} files not found. See missing_files.txt for details.")




scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2011/20110704-0900-PLENARY-14_fr_43.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2011/20110704-0900-PLENARY-14_fr_43.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2011/20110705-0900-PLENARY-12_fr_64.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2011/20110705-0900-PLENARY-12_fr_64.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2011/20110705-0900-PLENARY-12_fr_65.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2011/20110705-0900-PLENARY-12_fr_65.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2011/20110217-0900-PLENARY-4_fr_40.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2011/20110217-0900-PLENARY-4_fr_40.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2011/20110509-0900-PLENARY-14_fr_110.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2011/20110509-0900-PLENARY-14_fr_110.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090218-0900-PLENARY-20_fr_40.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090218-0900-PLENARY-20_fr_40.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090218-0900-PLENARY-20_fr_41.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090218-0900-PLENARY-20_fr_41.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090218-0900-PLENARY-20_fr_42.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090218-0900-PLENARY-20_fr_42.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090218-0900-PLENARY-20_fr_43.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090218-0900-PLENARY-20_fr_43.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090218-0900-PLENARY-20_fr_45.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090218-0900-PLENARY-20_fr_45.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090311-0900-PLENARY-16_fr_84.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090311-0900-PLENARY-16_fr_84.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090311-0900-PLENARY-16_fr_85.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090311-0900-PLENARY-16_fr_85.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090325-0900-PLENARY-15_fr_22.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090325-0900-PLENARY-15_fr_22.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090421-0900-PLENARY-16_fr_17.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090421-0900-PLENARY-16_fr_17.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2009/20090507-0900-PLENARY-7_fr_10.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090507-0900-PLENARY-7_fr_10.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2010/20100324-0900-PLENARY-10_fr_25.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2010/20100324-0900-PLENARY-10_fr_25.wav
scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2010/20100421-0900-PLENARY-4_fr_58.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2010/20100421-0900-PLENARY-4_fr_58.wav

scp /corpus/LeBenchmark/voxpopuli_unlabelled/wav/2014/20140114-0900-PLENARY-5_fr_6.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/wav/2014/20140114-0900-PLENARY-5_fr_6.wav



/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Voxpopuli_unlabeled_fr/wav/2009/20090218-0900-PLENARY-20_fr_41.wav

/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040117_0455_0940_RFI_ELDA_part1_s_308.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040117_0455_0940_RFI_ELDA_part1_s_308.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040117_0455_0940_RFI_ELDA_part1_s_308.wav




scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040324_1455_0105_INFO_ELDA_part1_s_476.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040324_1455_0105_INFO_ELDA_part1_s_476.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040328_1455_0105_INFO_ELDA_part12_s_39.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040328_1455_0105_INFO_ELDA_part12_s_39.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040316_0455_0905_INTER_ELDA_part3_s_220.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040316_0455_0905_INTER_ELDA_part3_s_220.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040327_1455_0105_INFO_ELDA_part11_s_76.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040327_1455_0105_INFO_ELDA_part11_s_76.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040420_0000_1159_RFI_ELDA_part18_s_269.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040420_0000_1159_RFI_ELDA_part18_s_269.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040120_0125_0240_RFI_ELDA_part2_s_93.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040120_0125_0240_RFI_ELDA_part2_s_93.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040317_1455_0105_INFO_ELDA_part9_s_264.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040317_1455_0105_INFO_ELDA_part9_s_264.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040322_1455_0105_INFO_ELDA_part10_s_93.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040322_1455_0105_INFO_ELDA_part10_s_93.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040921_0655_0905_CULTURE_ELDA_part1_s_244.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040921_0655_0905_CULTURE_ELDA_part1_s_244.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040429_0000_1159_RFI_ELDA_part8_s_291.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040429_0000_1159_RFI_ELDA_part8_s_291.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040819_0455_0905_INTER_ELDA_part4_s_41.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040819_0455_0905_INTER_ELDA_part4_s_41.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040320_0855_1005_INTER_ELDA_part1_s_247.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040320_0855_1005_INTER_ELDA_part1_s_247.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040328_1455_0105_INFO_ELDA_part4_s_101.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040328_1455_0105_INFO_ELDA_part4_s_101.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040320_0455_1505_INFO_ELDA_part7_s_161.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040320_0455_1505_INFO_ELDA_part7_s_161.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040117_0455_0940_RFI_ELDA_part1_s_308.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040117_0455_0940_RFI_ELDA_part1_s_308.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040430_0455_1456_INFO_DGA_part2_s_269.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040430_0455_1456_INFO_DGA_part2_s_269.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040330_0455_1505_INFO_ELDA_part11_s_229.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040330_0455_1505_INFO_ELDA_part11_s_229.wav
scp /corpus/LeBenchmark/epac_flowbert/EPAC_flowbert/output_waves/20040503_1155_1335_CULTURE_ELDA_part3_s_268.wav uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/EPAC_flowbert/output_waves/20040503_1155_1335_CULTURE_ELDA_part3_s_268.wav











scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_DIA_1223_2.978_4.893.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_DIA_1223_2.978_4.893.wav


scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_DIA_1223_2002.487_2004.18.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_DIA_1223_2002.487_2004.18.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_DIA_1223_2006.306_2008.734.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_DIA_1223_2006.306_2008.734.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_DIA_1223_2026.064_2027.343.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_DIA_1223_2026.064_2027.343.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_DIA_1226_4571.804_4573.275.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_DIA_1226_4571.804_4573.275.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2324.355_2325.992.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2324.355_2325.992.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2338.681_2339.72.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2338.681_2339.72.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2343.1_2345.542.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2343.1_2345.542.wav 

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2348.747_2349.899.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2348.747_2349.899.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2357.757_2359.432.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2357.757_2359.432.wav

scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_237.284_240.068.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_237.284_240.068.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2373.736_2375.777.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2373.736_2375.777.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2383.176_2384.545.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2383.176_2384.545.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2399.967_2401.396.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2399.967_2401.396.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_24.676_25.944.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_24.676_25.944.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_243.464_245.676.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_243.464_245.676.wav 
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2439.631_2440.86.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2439.631_2440.86.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2444.356_2446.301.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2444.356_2446.301.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2460.704_2462.254.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2460.704_2462.254.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2464.481_2466.321.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2464.481_2466.321.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2469.282_2470.754.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2469.282_2470.754.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ECOLE_1289_2476.737_2480.384.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_2476.737_2480.384.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ENT_1058_2907.32_2909.767.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ENT_1058_2907.32_2909.767.wav
scp /corpus/LeBenchmark/eslo2_train1_flowbert/ESLO/wav_turns/ESLO2_ENT_1058_2917.0_2918.379.wav \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ENT_1058_2917.0_2918.379.wav


/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ENT_1058_2907.32_2909.767.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/ESLO2/ESLO/wav_turns/ESLO2_ECOLE_1289_237.284_240.068.wav






