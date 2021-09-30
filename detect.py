"""
Before detecting an object, you must train a model first.

python C:/Python/models/research/object_detection/legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=config/ssd_mobilenet_v1_pets.config
tensorboard --logdir="training/"

python C:\Python\models\research\object_detection\export_inference_graph.py
--input_type=image_tensor
--pipeline_config_path=config\ssd_mobilenet_v1_pets.config
--trained_checkpoint_prefix=training/model.ckpt-10000
--output_directory=model
"""


import cv2
import numpy
import tensorflow
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
import argparse
import time
import threading
import queue
import datetime


"""
queue is a line for thread to respect
current is which image should output now
"""
id_queue = 0
current = 0
image_queue = queue.Queue(maxsize=9000)
total_recog_time = 0
FrameExist = True
showing = False


class Detector(object):
    def __init__(self):
        self.PATH_TO_CKPT = 'model/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = 'data/object-detection.pbtxt'
        self.NUM_CLASSES = 2
        self.detection_graph = self.load_model()
        self.category_index = self.load_label_map()

    def load_model(self):
        detection_graph = tensorflow.Graph()
        with detection_graph.as_default():
            od_graph_def = tensorflow.GraphDef()
            with tensorflow.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tensorflow.import_graph_def(od_graph_def, name='')
        return detection_graph

    def load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, thread_id, windowName, image):
        global current, total_recog_time, image_queue
        tStart = time.time()
        with self.detection_graph.as_default():
            with tensorflow.Session(graph=self.detection_graph) as session:
                # expand dimensions
                image_np_expanded = numpy.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # actual detection
                (boxes, scores, classes, num_detections) = session.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )

                # visualization of the result of a detection
                visualization_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    numpy.squeeze(boxes),
                    numpy.squeeze(classes).astype(numpy.int32),
                    numpy.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=10
                )

                #  cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
                i = None
                while current != thread_id:
                    i = None
                cv2.imshow(windowName, image)
                tEnd = time.time()
                print("The detection cost " + str(tEnd - tStart) + " seconds.")
                current += 1
                #  cv2.waitKey(0)


def main(args):
    global id_queue, FrameExist, total_recog_time
    str_path = args.pic_path
    """
    image = cv2.imread(str_path)
    ImgDetector = Detector()
    ImgDetector.detect(image)
    """
    video = cv2.VideoCapture('testing_object/amber.mp4')
    windowName = 'Video Detection'
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    ImgDetector = Detector()
    if video.isOpened():
        while True:
            ret, prev = video.read()
            if ret:
                #  cv2.imshow('Video', prev)
                threading.Thread(target=ImgDetector.detect, args=(id_queue, windowName, prev), daemon=True).start()
                id_queue += 1
                if (id_queue - current) > 16:
                    time.sleep(12)
            else:
                break
            if 0xFF & cv2.waitKey(1) == 27:
                break
    FrameExist = False
    cv2.destroyAllWindows()
    print("Average recognition time per frame = ", str(total_recog_time/id_queue))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Need to input the path of target picture.")
    parser.add_argument('--pic_path', '-p', help='The path of target picture.', required=True)
    main(parser.parse_args())
