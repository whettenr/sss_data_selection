/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/African_Accented_French/wavs/train/ca16/079/afc-gabon_16.06.16_079_read_0662.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/African_Accented_French/wavs/test/ca16/chad_m_001/chad_m_001_001.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/African_Accented_French/wavs/train/ca16/083/afc-gabon_16.06.16_083_read_0284.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/GEMEP/wavs/W08pla412.wav

/corpus/LeBenchmark/gemep_flowbert/GEMEP/wavs/W08pla412.wav

/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Att-HACK_SLR88/wavs/M12_a1_s032_v01.wav
/corpus/LeBenchmark/slr88_flowbert/Att-HACK_SLR88/wavs/M12_a1_s032_v01.wav


/corpus/LeBenchmark/slr57_flowbert/African_Accented_French/wavs/test/ca16/chad_m_001/chad_m_001_001.wav

rsync -av --checksum --progress --delete \
    /corpus/LeBenchmark/slr57_flowbert/African_Accented_French/wavs/ \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/African_Accented_French/wavs


rsync -av --checksum --progress --delete \
    /corpus/LeBenchmark/slr88_flowbert/Att-HACK_SLR88/wavs/ \
    uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Att-HACK_SLR88/wavs


rsync -av --checksum --progress --delete \
    /corpus/LeBenchmark/slr88_flowbert/Att-HACK_SLR88/wavs/ uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Att-HACK_SLR88/wavs

rsync -av --checksum --progress --delete \
    /corpus/LeBenchmark/CFPP_corrected/CFPP_corrected/output/ uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/CFPP_corrected/output

rsync -av --checksum --progress --delete \
    /corpus/LeBenchmark/gemep_flowbert/GEMEP/wavs/ uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/GEMEP/wavs

rsync -av --checksum --progress --delete \
    /corpus/LeBenchmark/port_media_flowbert/PMDOM2FR_00/PMDOM2FR_wavs/ uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Portmedia/PMDOM2FR_wavs


rsync -av --checksum --progress --delete \
    /corpus/LeBenchmark/TCOF_corrected/TCOF_corrected/output/ uaj64gk@jean-zay.idris.fr:/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/TCOF_corrected/output


/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Portmedia/PMDOM2FR_wavs/BLOCK12/wavs_turns/08730_1250_215.4_216.18.wav
/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/TCOF_corrected/output/coree_ghu_14_s_544.wav


/lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/GEMEP/wavs/W08pla412.wa

#!/bin/bash
#SBATCH --job-name=rtouch
#SBATCH --account=dha@cpu
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=log/rtouch-%j.log
#SBATCH --ntasks-per-node=1


DATE=$(date +%Y%m%d-%H%M%S)
cd /lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Att-HACK_SLR88
find /lustre/fsn1/projects/rech/nkp/uaj64gk/LeBenchmark/all_outputs/Att-HACK_SLR88 -exec touch {} +

mv log/rtouch-${SLURM_JOB_ID}.log log/rtouch-$DATE.log