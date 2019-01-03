import tensorflow as tf

# Assuming object detection API is available for use
from object_detection.utils.config_util import create_pipeline_proto_from_configs
from object_detection.utils.config_util import get_configs_from_pipeline_file
import object_detection.exporter

# Configuration for model to be exported
config_pathname = r'/Users/wudan/Desktop/tensorflow_file/models/research/object_detection/WB_Car_New/config/ssd_mobilenet_v1_coco.config'

# Input checkpoint for the model to be exported
# Path to the directory which consists of the saved model on disk (see above)
trained_model_dir = r'/Users/wudan/Desktop/tensorflow_file/models/research/object_detection/WB_Car_New/training/model.ckpt-73049'

# Create proto from model confguration
configs = get_configs_from_pipeline_file(config_pathname)
pipeline_proto = create_pipeline_proto_from_configs(configs=configs)

# Read .ckpt and .meta files from model directory
# checkpoint = tf.train.get_checkpoint_state(trained_model_dir)
# input_checkpoint = checkpoint.model_checkpoint_path

# input_type,
#                            pipeline_config,
#                            trained_checkpoint_prefix,
#                            output_directory,
#                            input_shape=None,
#                            output_collection_name='inference_op',
#                            additional_output_tensor_names=None,
#                            write_inference_graph=False

# Model Version
model_version_id = 43049

# Output Directory
output_directory = r'/Users/wudan/Desktop/tensorflow_file/models/research/object_detection/WB_Car_New/model/'+ str(model_version_id)

# Export model for serving
object_detection.exporter.export_inference_graph(input_type='image_tensor',pipeline_config=pipeline_proto,trained_checkpoint_prefix=trained_model_dir,output_directory=output_directory)
