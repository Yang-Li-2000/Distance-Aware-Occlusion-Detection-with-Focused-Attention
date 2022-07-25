# Distance-aware Occlusion Detection with Focused Attention
Code for Distance-Aware Occlusion Detection with Focused Attention


## Environment and Data
### 1. install dependencies
```bash
conda create --name ENVIRONMENT_NAME python=3.8.10
conda activate ENVIRONMENT_NAME
pip install -r requirements.txt
```



### 2. download and prepare data
The [2.5VRD dataset](https://github.com/google-research-datasets/2.5vrd) needs to be placed in the following way. The three .odgt 
files listed below need to be generated using our write annotations.ipynb. 

    Project Root/
        └── data/
            └── 2.5vrd/
                └──images/
                    ├── train/
                    ├── validation/
                    ├── test/
                    ├── annotation_train_combined.odgt
                    ├── annotation_valid_combined.odgt
                    ├── annotation_test_combined.odgt
                    └── write annotations.ipynb

### 3. download checkpoints (pre-trained models)
| Model                                                | Distance F1-Score  | Occlusion F1-Score | checkpoint                                                                                                                                                                                               |
|------------------------------------------------------|--------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GIT                                                  | 0.3857  | 0.4124             | [GIT.pth](https://www.icloud.com.cn/iclouddrive/09dGyXkFES8gdEQ9OIMRXReCg#GIT)                                                                                                                           |
| GIT, no intersection loss when no intersection exists | 0.3788 | 0.4050             | [GIT_do_not_calculate_intersection_loss_if_no_intersection_exists.pth](https://www.icloud.com.cn/iclouddrive/0e8es6CIwStS9FqjkTcgrDjkA#GIT_do_not_calculate_intersection_loss_if_no_intersection_exists) |
| without GIT                                          | 0.3710               | 0.3995             | [no_GIT.pth](https://www.icloud.com.cn/iclouddrive/0c0L7OD4W-u8Z2bSagnHqB07A#no_GIT)                                                                                                                                                                                           |



## Evaluate (on the test set)

The argument --resume is the path of the model checkpoint to be evaluated. Evaluation out will be written to tensorboard. The experiment name is the name of the folder that is under output_dir and contains the checkpoint. For example, the sample evaluation script would result in an experiment name of "Dec28_Cascade_633".\
Users need to modify dec_layers, dec_layers_distance, dec_layers_occlusion to the correct number of transformer decoder layers in the object pair decoder, distance decoder, and occlusion decoder, respectively. 

### 1. Generate and Save Predictions
#### a) model trained with the generalized intersection prediction task (GIT)

Before running, in the unmodified magic_numbers.py, set:\
**PREDICT_INTERSECTION_BOX = True**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/GIT/GIT.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```


#### b) model trained without the generalized intersection prediction task (no GIT)
Before running, in the unmodified magic_numbers.py, make sure:\
**PREDICT_INTERSECTION_BOX = False**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/no_GIT/no_GIT.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```

After generating predictions, need to evaluate them and write evaluation outputs to tensorboard.

### 2. Evaluate using saved predictions 
Use the evaluation scripts provided by the authors of [2.5VRD](https://github.com/google-research-datasets/2.5vrd) to evaluate the performance of our models. 




## Train
The experiment_name, output_dir, dec_layers, dec_layers_distance, dec_layers_occlusion arguments can be changed.
1. experiment_name: the experiment name in tensorboard.
2. output_dir: the folder to store checkpoints.
3. dec_layers, dec_layers_distance, dec_layers_occlusion corresponds to the number of transformer decoder layers in the object pair decoder, distance decoder, and occlusion decoder, respectively. 

### 1. Train with the generalized intersection prediction task (GIT)
Before running, in the unmodified magic_numbers.py, set:\
**PREDICT_INTERSECTION_BOX = True**

```bash
# 8 GPUs (40G GPU memory per GPU)
torchrun --nnodes=1 --nproc_per_node=8 --master_port=54321 main.py --num_workers=8 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=6 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
# 4 GPUs (80G GPU memory per GPU)
torchrun --nnodes=1 --nproc_per_node=4 --master_port=54321 main.py --num_workers=4 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=12 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
```

### 2. Train without the generalized intersection prediction task (no GIT)
Before running, in the unmodified magic_numbers.py, make sure:\
**PREDICT_INTERSECTION_BOX = False**

```bash
# 8 GPUs (40G GPU memory per GPU)
torchrun --nnodes=1 --nproc_per_node=8 --master_port=54322 main.py --num_workers=8 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=6 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
# 4 GPUs (80G GPU memory per GPU)
torchrun --nnodes=1 --nproc_per_node=4 --master_port=54322 main.py --num_workers=4 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=12 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
```

### 3. Use a single decoder for all tasks
```bash
# 8 GPUs (40G GPU memory per GPU)
torchrun --nnodes=1 --nproc_per_node=8 --master_port=54323 main.py --num_workers=8 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=6 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
# 4 GPUs (80G GPU memory per GPU)
torchrun --nnodes=1 --nproc_per_node=4 --master_port=54323 main.py --num_workers=4 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=12 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
```


## Visualize Attention Weights (on the test set)

### 1. Save attention weights to disk
Firstly, set **SORT_USING_OBJECT_SCORES = True** in magic_numbers.py


#### a) Model trained with the GIT
Before running, in the unmodified magic_numbers.py, set:\
**VISUALIZE_ATTENTION_WEIGHTS = True**\
**PREDICT_INTERSECTION_BOX = True**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/GIT/GIT.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```
Running this would save the attention weights of the model that was trained with the generalized intersection prediction task to disk.

#### b) model trained without the GIT
Before running, in the unmodified magic_numbers.py, set:\
**VISUALIZE_ATTENTION_WEIGHTS = True**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/no_GIT/no_GIT.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```
Running this would save the attention weights of the model trained without the generalized intersection prediction task to disk.

### 2. Visualize saved attention weights using jupyter notebooks
Use a jupyter notebook provided by us to visualize attention weights saved to disk in the previous steps. 

Place this notebooks under project root and follow the instructions in it to visualize decoder attentions:\
[cleaned_visualize_attention_(GIT).ipynb]()


## Debug
```bash
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --num_workers=0 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=6 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
```

## Citation

```
TODO
```
