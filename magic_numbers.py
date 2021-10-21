# Train on a specific image specified by the index of that image
TRAIN_ON_ONE_IMAGE = False
index_of_that_image = 2

# Train or validate on a subset of the complete dataset.
USE_SMALL_ANNOTATION_FILE = False
small_annotation_file = 'small_train_combined.odgt'
USE_SMALL_VALID_ANNOTATION_FILE = False
small_valid_annotation_file = 'small_valid_combined.odgt'

# Disable shuffle or not. Useful for debugging.
USE_SEQUENTIAL_LOADER = False

# Produce debug outputs or not.
DEBUG_OUTPUTS = False
top_k_predictions_to_print = 10

# Whether to test whether the cost matrix for optimal trasport is correctly computed.
TEST_COST_MATRIX = False


# Thresholds above which generate_hoi_list_using_model_outputs() will add a hoi to hoi_list
human_th=0
object_th=0
hoi_th=0
occlusion_th=0

# nms thresholds in triplet_nms_for_vrd()
nms_iou_human = 0.7
nms_iou_object = 0.7

# threshold in generate_hoi_list_using_model_outputs() when filter=True
human_th_debug = 0
object_th_debug = 0
hoi_th_debug = 0
occlusion_th_debug = 0


num_workers_train = 16 # Not implemented yet
batch_size_train = 5 # Not implemented yet


# num_workers and batch size for the validation and test sets
num_workers_validation = 2 # 16
batch_size_validation = 30  # 30


# Whether to use optimal transport. If not, use the Hungarian matcher.
USE_OPTIMAL_TRANSPORT = True
# k in optimal transport
OT_k = 10


BACK_PROP_SINKHORN_COST = False