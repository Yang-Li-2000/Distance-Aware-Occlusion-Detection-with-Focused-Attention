sharing_strategy = "file_descriptor"
#sharing_strategy = "file_system"

# Train on a specific image specified by the index of that image
TRAIN_ON_ONE_IMAGE = False
index_of_that_image = 2

# Train or validate on a subset of the complete dataset.
USE_SMALL_ANNOTATION_FILE = False
small_annotation_file = 'small_train_combined.odgt'
#small_annotation_file = 'medium_train_combined.odgt'
USE_SMALL_VALID_ANNOTATION_FILE = False
small_valid_annotation_file = 'small_valid_combined.odgt'
#small_valid_annotation_file = 'small_valid_combined_custom.odgt'
USE_SMALL_TEST_ANNOTATION_FILE = False
small_test_annotation_file = 'small_test_combined.odgt'

# Disable shuffle or not. Useful for debugging.
USE_SEQUENTIAL_LOADER = False
# Produce debug outputs or not.
DEBUG_OUTPUTS = False
top_k_predictions_to_print = 10
SAVE_IMAGES = False
# Whether to test whether the cost matrix for optimal trasport is correctly computed.
TEST_COST_MATRIX = False

# LR Range Test
LR_RANGE_TEST = False
CYCLIC_SCHEDULER = False
CYCLIC_BASE_LR = 0.000007
CYCLIC_MAX_LR = 0.000035
CYCLIC_STEP_SIZE_UP = 5
CYCLIC_STEP_SIZE_DOWN = 5

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

# Whether to use optimal transport. If not, use the Hungarian matcher.
USE_OPTIMAL_TRANSPORT = False
OT_k = 8 # k in optimal transport
# Parameters for SinkhornDistance
SINKHORN_MAX_ITER_eps = 0.01 # default is 1e-3
SINKHORN_MAX_ITER = 1000 # default is 100
# Extra parameters of matchers
BACK_PROP_SINKHORN_COST = False
USE_DYNAMIC_K_ESTIMATE = False
HUNGARIAN_K_ASSIGNMENTS = False
NORMALIZED_MAX = False
BG_COEF = 0.0

# Persistent worker for dataloader_train when num_workers > 0
PERSISTENT_WORKERS=True
# num_workers and batch size for the validation and test sets
num_workers_validation = 8 # 16
batch_size_validation = 10  # 30

# Uses the maximum resolution, instead of randomly select from scales,
# to test if GPU memory is enough for training
GPU_MEMORY_PRESSURE_TEST = False

# Attention Visualization
VISUALIZE_ATTENTION_WEIGHTS = False

# Cascade Decoders
CASCADE = True
# Add depth to positional encodings
USE_DEPTH_DURING_TRAINING = False
USE_DEPTH_DURING_INFERENCE = False
# Predict Intersection Box
PREDICT_INTERSECTION_BOX = False
# Use raw labels as targets
USE_RAW_DISTANCE_LABELS = False
USE_RAW_OCCLUSION_LABELS = False

IMPROVE_INTERMEDIATE_LAYERS = False
