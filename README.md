# DualText

1. Setup the conda environment 
    ```
    conda create --name dualtext python=3.9
    conda activate dualtext
    # install the pytorch version compatible with the your cuda version
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install -r requirements.txt
    ```
2. Download MANO model files (`MANO_LEFT.pkl` and `MANO_RIGHT.pkl`) from the [website](https://mano.is.tue.mpg.de/) and place them in the `tool/mano_models` folder.

3. Download the YCB models from [here](https://rse-lab.cs.washington.edu/projects/posecnn/) and set `object_models_dir` in `config.py` to point to the dataset folder. The original mesh models are large and have different vertices for different objects. To enable batched inference, we additionally use simplified object models with 1000 vertices. Download the simplified models from [here](https://zenodo.org/records/11668766) and set `simple_object_models_dir` in `config.py` to point to the dataset folder

4. Download the processed annotation files for both datasets from [here](https://zenodo.org/records/11668766) and set `annotation_dir` in `config.py` to point to the processed data folder.

## Dataset Setup
Depending on the dataset you intend to train/evaluate follow the instructions below for the setup.

### HO3Dv2 Setup
1. Download the dataset from the [website](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/) and set `ho3d_data_dir` in `config.py` to point to the dataset folder.
2. Obtain Signed-Distance-Field (SDF) files for every sample. This is only needed for the training set. You can obtain them in either of the following ways. Set `fast_data_dir` in `config.py` to point to the processed SDF folder.
    * Download the processed SDF files for HO3Dv2 training set from [here](https://zenodo.org/records/13228003).
    * Then use the `tool/pre_process_sdf.py` script to process the SDF data.
### DexYCB Setup
1. Download the dataset from the [website](https://dex-ycb.github.io/) and set `dexycb_data_dir` in `config.py` to point to the dataset folder.
2. Obtain Signed-Distance-Field (SDF) files for every sample. This is needed for both the training and test sets. You can obtain them in either of the following ways. Set `fast_data_dir` in `config.py` to point to the processed SDF folder.
    * Download the processed SDF files for DexYCB test set from [here](https://drive.google.com/drive/folders/1GoaA6vB6TwAAHmaobVo5GjRoCq2wT21R). 
    * Then use the `tool/pre_process_sdf.py` script to process the SDF data.

## Text Prompt Setup
1. In order to accelerate the running speed of the code, use `data/pre_process/pre_process_oog.py` script to obtain the Object name, Occlusion label, and Grasping state, and save them in `YCB_Train_OGG.json`.
   * We have provided the relevant demo data in `data/demo/Demo_DexYCB_Train.json`, and its specific storage format follows the official dataset.

2. Use `data/pre_process/pre_process_text.py` to obtain the dual-scale structured text prompt and the corresponding text features.
   * During the operation, the relevant paths need to be modified to your own real path.

## Evaluation
Depending on the dataset you intend to evaluate follow the instructions below. 

### HO3Dv2
1. In the `config.py`, modify `setting` parameter.
    * `setting = 'ho3d'` for evaluating the model only trained on the HO3Dv2 training set.

2. Run the following command:
    ```
    python main/test.py --ckpt_path ckpts/ho3d/snapshot_ho3d.pth.tar  # for ho3d setting
    ```
3. The results are dumped into a `results.txt` file in the folder containing the checkpoint.
 
#### DexYCB
1. In the `config.py`, modify `setting` parameter.
    * `setting = 'dexycb'` for evaluating the model only trained on the DexYCB split, which only includes the right hand data.
2. Run the following command:
    ```
    python main/test.py --ckpt_path ckpts/dexycb/snapshot_dexycb.pth.tar  # for dexycb setting
    ```
3. The results are dumped into a `results.txt` file in the folder containing the checkpoint.

## Training
Depending on the dataset you intend to train follow the instructions below.

1. Set the `output_dir` in `config.py` to point to the directory where the checkpoints will be saved.
2. In the `config.py`, modify `setting` parameter.
    * `setting = 'ho3d'` for training the model on the HO3Dv2-split training set.
    * `setting = 'dexycb'` for training the model on the DexYCB, which only includes the right hand data.
3. Run the following command, set the `CUDA_VISIBLE_DEVICES` and `--gpu` to the desired GPU id. Here is an example command for training on one GPU:
    ```
    CUDA_VISIBLE_DEVICES=0 python main/train.py --run_dir_name test --gpu 0
    ```
4. To continue training from the last saved checkpoint, add `--continue` argument in the above command.
3. The checkpoints are dumped after every epoch in the 'output' folder of the base directory.
4. Tensorboard logging is also available in the 'output' folder.


## Acknowlegements

* Some of the code has been reused from [HOISDF](https://github.com/amathislab/HOISDF/tree/main?tab=readme-ov-file), [Text-IF](https://github.com/XunpengYi/Text-IF) and [LAMP](https://github.com/shengnanh20/LAMP) repositories. We thank the authors for sharing their excellent work!

