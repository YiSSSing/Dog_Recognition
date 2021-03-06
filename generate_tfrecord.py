"""
from tensorflow/models/

create train data :
python generate_tfrecord.py --csv_input=data/train_labels.csv --output_path=data/train.record --image_dir=resources_img/

create test data :
python generate_tfrecord.py --csv_input=data/test_labels.csv --output_path=data/test.record --image_dir=resources_img/
"""

from __future__ import division
from __future__ import absolute_import

import os
import io
import pandas
import tensorflow
__all__ = [tensorflow]

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple


flags = tensorflow.app.flags
flags.DEFINE_string('csv_input', 'data', 'Path to the CSV input')
flags.DEFINE_string('output_path', 'data', 'Path to output the TFRecord')
flags.DEFINE_string('image_dir', 'images', 'Path to images')
FLAGS = flags.FLAGS


def class_to_id(row_label):
    if row_label == 'dog':
        return 1
    elif row_label == 'cat':
        return 2
    else:
        return None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tfrecord(group, path):
    with tensorflow.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_to_id(row['class']))

    tfrecord = tensorflow.train.Example(features=tensorflow.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tfrecord


def main():
    writer = tensorflow.io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    records = pandas.read_csv(FLAGS.csv_input)
    grouped = split(records, 'filename')
    for group in grouped:
        tfrecord = create_tfrecord(group, path)
        writer.write(tfrecord.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully create TF records: {}'.format(output_path))


if __name__ == '__main__':
    tensorflow.compat.v1.app.run()
