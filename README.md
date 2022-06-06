# Distance-aware Occlusion Detection from Pair Proposals with Focused Attention
Code for TODO


## Performance
TODO

## (TODO) Environment and Data
conda activate yang_hoitr


## Evaluate (on the test set)

The argument --resume is the path of the model checkpoint to be evaluated. Evaluation out will be written to tensorboard. The experiment name is the name of the folder that is under output_dir and contains the checkpoint. For example, the sample evaluation script would result in an experiment name of "Dec28_Cascade_633".\
Users need to modify dec_layers, dec_layers_distance, dec_layers_occlusion to the correct number of transformer decoder layers in the object pair decoder, distance decoder, and occlusion decoder, respectively. 

### 1. Generate and Save Predictions
#### a) model trained with the generalized intersection prediction task (GIT)

Before running, in the unmodified magic_numbers.py, set:\
**PREDICT_INTERSECTION_BOX = True**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/Dec27_Cascade_633+Intersection/checkpoint_epoch_40.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```


#### b) model trained without the generalized intersection prediction task (GIT)
Before running, in the unmodified magic_numbers.py, make sure:\
**PREDICT_INTERSECTION_BOX = False**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/Dec28_Cascade_633/checkpoint_epoch_37.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```

After generating predictions, need to evaluate them and write evaluation outputs to tensorboard.

### 2. (TODO) Evaluate using saved predictions 
go to ../temp_eval_for_hoitr and run the python file named write_to_tensorboard.py




## Train
The experiment_name, output_dir, dec_layers, dec_layers_distance, dec_layers_occlusion arguments can be changed.
1. experiment_name: the experiment name in tensorboard.
2. output_dir: the folder to store checkpoints.
3. dec_layers, dec_layers_distance, dec_layers_occlusion corresponds to the number of transformer decoder layers in the object pair decoder, distance decoder, and occlusion decoder, respectively. 

### 1. Train with the generalized intersection prediction task (GIT)
Before running, in the unmodified magic_numbers.py, set:\
**PREDICT_INTERSECTION_BOX = True**

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=54321 main.py --num_workers=8 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=6 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
```

### 2. Train without the generalized intersection prediction task (GIT)
Before running, in the unmodified magic_numbers.py, make sure:\
**PREDICT_INTERSECTION_BOX = False**

```bash
torchrun --nnodes=1 --nproc_per_node=8 --master_port=54322 main.py --num_workers=8 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=6 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
```

## Visualize Attention Weights (on the test set)

### 1. Save attention weights to disk
Firstly, set **SORT_USING_OBJECT_SCORES = True** in magic_numbers.py


#### a) Model trained with the GIT
Before running, in the unmodified magic_numbers.py, set:\
**VISUALIZE_ATTENTION_WEIGHTS = True**\
**PREDICT_INTERSECTION_BOX = True**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/Dec27_Cascade_633+Intersection/checkpoint_epoch_40.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```
Running this would save the attention weights of the model that was trained with the generalized intersection prediction task to disk.

#### b) model trained without the GIT
Before running, in the unmodified magic_numbers.py, set:\
**VISUALIZE_ATTENTION_WEIGHTS = True**
```bash
python vrd_test.py --backbone=resnet101 --resume='output_dir/Dec28_Cascade_633/checkpoint_epoch_37.pth' --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --num_workers=0 --batch_size=1
```
Running this would save the attention weights of the model trained without the generalized intersection prediction task to disk.

### 2. Visualize saved attention weights using jupyter notebooks
(TODO) Use the a jupyter notebook provided by us to visualize attention weights saved to disk in the previous steps. 

Copy and use these two notebooks:\
70: /DATA1/liyang/HoiTransformer/visualize_attention_intersection.ipynb\
70: /DATA1/liyang/HoiTransformer/visualize_attention.ipynb

## Debug
```bash
CUDA_VISIBLE_DEVICES=0 python -m pdb main.py --num_workers=0 --epochs=500 --dataset_file=two_point_five_vrd --batch_size=6 --backbone=resnet101 --lr=0.0001  --dec_layers=6 --dec_layers_distance=3 --dec_layers_occlusion=3 --experiment_name='runs/debug'  --output_dir='output_dir/debug' --lr_drop=30
```


## Citation

```
TODO
```


## Acknowledgement
TODO