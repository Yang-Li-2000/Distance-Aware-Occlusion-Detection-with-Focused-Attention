# Train on a specific image
TRAIN_ON_ONE_IMAGE = False
index_of_that_image = 2

# Train on a subset of the complete dataset.
USE_SMALL_ANNOTATION_FILE = False
small_annotation_file = 'small_train_combined.odgt'

# Disable shuffle or not. Useful for debugging.
USE_SEQUENTIAL_LOADER = False

# Produce debug outputs or not.
DEBUG_OUTPUTS = False
top_k_predictions_to_print = 10


# Thresholds above which generate_hoi_list_using_model_outputs() will add a hoi to hoi_list
human_th=0
object_th=0
hoi_th=0
occlusion_th=0

# nms thresholds in triplet_nms_for_vrd()
nms_iou_human = 0.7
nms_iou_object = 0.7


num_workers_train = 16 # Not implemented yet
batch_size_train = 5 # Not implemented yet


num_workers_validation = 8 # 48
batch_size_validation = 30  # 30