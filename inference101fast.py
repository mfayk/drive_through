import numpy as np
import os
import tensorflow as tf

from python.modules.utils.data_utils import better_makedirs
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('model_dir',
					'data/checkpoints/ssd_mobilenet_v2_01',
					'Frozen graph directory.')
flags.DEFINE_string('labelmap_path',
					'data/ua_detrac_labelmap.pbtxt',
					'Labelmap path. Defaults to \'data/ua_detrac_labelmap.pbtxt\'')
flags.DEFINE_string('test_data_dir',
					'/media/yingges/TOSHIBA EXT/datasets/trafficvision/UADETRAC/Insight-MVT_Annotation_Test',
					'Test images dir.')
flags.DEFINE_string('video_name',
					'MVI_40714',
					'Which video sequence to run inference on.')
flags.DEFINE_string('output_dir',
					'data/inference_output',
					'Defaults to \'data/inference_output\'')
flags.DEFINE_integer('classnum', 
					 4,
					 'Number of classes.')
FLAGS = flags.FLAGS

imageSize = (540, 960)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          print("in tensor name")
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
       	detection_boxes = tf.squeeze(tensor_dict['detection_boxes'],[0])
        print("in tensor dict/n")
        detection_masks = tf.squeeze(tensor_dict['detection_masks'],[0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      print("image: "),
      print(image_tensor)
      print("end")
      # Run inference with 5 im good
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})#np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
	#graph can run for muti
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
      return output_dict

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  #print(image.size)
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main(_):
	fg_path = os.path.join(FLAGS.model_dir, 'frozen_inference_graph.pb')
	print(fg_path)
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(fg_path, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	label_map = label_map_util.load_labelmap(FLAGS.labelmap_path)
	categories = label_map_util.convert_label_map_to_categories(
					label_map, max_num_classes=FLAGS.classnum, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	video_name = FLAGS.video_name
	video_dir = os.path.join(FLAGS.test_data_dir, video_name)
	test_img_paths = [os.path.join(video_dir, im) 
						for im in os.listdir(video_dir)]
	# test_img_paths = []
	output_dir = os.path.join(FLAGS.output_dir, video_name)
	better_makedirs(output_dir)
	fhand = open('detection_file_resnet50_fast.txt','a')
	count = 0
	np.set_printoptions(threshold=np.nan)
	imagevalues = []
	#graphvalues = []
	print(type(imagevalues))
	for imfile in test_img_paths:
		#here change the count to effect the speed if the detections
		#if(count%5 != 0):
			print("count: "),
			print(count)
			if(count == 25):
				break
			c1 = imfile.rsplit("/")[-1].split(".")[0][3:]
			image = Image.open(imfile)
			image = load_image_into_numpy_array(image)
			#print(type(imagevalues))
			##print(type(image))
			#print(type(detection_graph))
			imagevalues.append(image)
			#print(imagevalues)
		#	graphvalues = np.append(graphvalues,detection_graph)
			#print("image")
			#print(image)
			#print("detection_graph")
			#print(detection_graph)
			#print(type(imagevalues))
			count = count+1		



	imagevalues = np.asarray(imagevalues)
	print(imagevalues.shape)
	print(type(imagevalues[0]))
	output_dict = run_inference_for_single_image(imagevalues, detection_graph)	
	img_scale = [imageSize[0], imageSize[1], imageSize[0], imageSize[1]]
	num=0
	#print(output_dict)
	for d in output_dict['detection_boxes']:
		if output_dict['detection_scores'][num] < 0.1:
			num = num +1
			continue 
		#print(d)
		#print(d[2])
		bbox = [i*j for i,j in zip(img_scale, d)]
		d_f = [int(c1),-1,bbox[1],bbox[0],bbox[3]-bbox[1],bbox[2]-bbox[0],output_dict['detection_scores'][num],-1,-1]
		num = num+1
		#print(bbox[2])
		fhand.write(str(str(d_f)[1:-1]+'\n'))
	



	fhand.close()

if __name__ == '__main__':
	tf.app.run()

