
#  Dynamic Geometry Reconstruction

<img width="1145" alt="overview" src="https://github.com/user-attachments/assets/47715677-539b-455c-893a-832c280a53bd" />

https://github.com/user-attachments/assets/0b323c85-d92b-40df-ad87-381b25572ade

## Environment Setup
Please follow the [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) to install the relative packages.
```bash
git clone git@github.com:buma13/E-D3DGS.git
cd E-D3DGS
git submodule update --init --recursive

conda create -n dynamic python=3.7 
conda activate dynamic

pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# If submodules fail to be downloaded, refer to the repository of 3DGS  
pip install -r requirements.txt

conda install nvidia/label/cuda-11.8.0::cuda-nvcc
conda install nvidia/label/cuda-11.8.0::cuda-cudart-dev
conda env config vars set CUDA_HOME=$CONDA_PREFIX

pip install -e submodules/diff-gaussian-rasterization/
pip install -e submodules/simple-knn/
```

## Data Preparation

**Downloading Datasets:**  
Please download dataset from its official website: [NeRSemble](https://tobias-kirschstein.github.io/nersemble/)



**Preparing dataset for E-D3DGS data loader:**
``` bash
python nersemble_scripts/prepare_nersemble_4_ed3dgs.py $RAW_GT_PATH/$PERSON $SCENE $PROCESSED_GT_PATH --alpha_mask --number_of_frames $SIZE
```

**Extracting point clouds from COLMAP:** 
```bash
# setup COLMAP 
bash script/colmap_setup.sh
conda activate colmapenv 

# automatically extract the frames and reorginize them
python script/pre_nersemble.py --videopath $PROCESSED_GT_PATH

# downsample dense point clouds
python script/downsample_point.py $PROCESSED_GT_PATH/colmap/dense/workspace/fused.ply $PROCESSED_GT_PATH/points3D_downsample.ply
```


After running COLMAP, Neural 3D Video and Technicolor datasets are orginized as follows:
```
├── data
│   | nersemble
│     ├── TONGUE
│       ├── colmap
│       ├── images
│           ├── cam01
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│           ├── cam02
│               ├── 0000.png
│               ├── 0001.png
│               ├── ...
│     ├── HAIR
|     ├── ...
``` 

**Script to help with adding tongue specific points to the point cloud:**
``` bash
# Renders point cloud with added tongue points and saves it to new file.
python nersemble_scripts/add_tongue_points.py $PROCESSED_GT_PATH/points3D_downsample.ply
``` 

## Training

To resize the training image, modify `-r 2` in the command line.
``` bash
# Train
python train.py -s $PROCESSED_GT_PATH/$SCENE --configs arguments/$DATASET/$CONFIG.py --model_path $OUTPUT_PATH --expname $EXPERIMENT -r 2
``` 

## Rendering

``` bash
# Render test view only
python render.py --model_path $OUTPUT_PATH --configs arguments/$DATASET/$CONFIG.py --skip_train --skip_video

# Render train view, test view, and spiral path
python render.py --model_path $OUTPUT_PATH --configs arguments/$DATASET/$CONFIG.py
```

## Mesh Extraction

``` bash
python mesh_extract_tetrahedra.py -s $PROCESSED_GT_PATH/$SCENE -m $OUTPUT_PATH -r 2 --configs arguments/$DATASET/$CONFIG.py --start_timestep_index $START_INDEX --end_timestep_index $STOP_INDEX
```

## Evaluation

``` bash
# Evaluate rendering metrics
python metrics.py --model_path $SAVE_PATH/$DATASET/$CONFIG

# Evaluate mesh metrics
python evaluate_pointcloud_mesh.py --meshes_path $OUTPUT_PATH/tetrahedra_meshes/ours_80000/ --scene_path $PROCESSED_GT_PATH/$SCENE --start_timestep_index $START_INDEX --end_timestep_index $STOP_INDEX
```

## Acknowledgements

This code is based on [E-D3DGS](https://github.com/JeongminB/E-D3DGS) and [RaDe-GS](https://github.com/BaowenZ/RaDe-GS). We would like to thank the authors of these papers for their hard work. 😊

