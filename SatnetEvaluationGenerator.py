from EvaluationGenerator import EvaluationGenerator
import os
import io
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import hashlib
from hpc_tensorflow.satnet_study.models.research.object_detection.utils import dataset_util
from hpc_tensorflow.satnet_study.models.research.object_detection.astro_net_evaluate_batch import AstroNetEvaluateBatch

FLAGS = None

CLASS_TEXT = ['None', 'Satellite']


def get_files(directory, extension, recursive=False):
    file_list = []
    for main_dir, sub_directories, _ in os.walk(directory):
        for file in os.listdir(main_dir):
            if file.endswith(extension):
                file_list.append(os.path.normpath(os.path.join(main_dir,
                                                               file)))
        if not recursive:
            return file_list
    return file_list


def load_annotations(source_directory):
    annotations = list()
    directories = sorted([f for f in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, f))])
    for f in tqdm(directories):
        annotations_path = os.path.join(source_directory, f, 'Annotations')

        # get the list of annotation files in the Annotations directory:
        annotations_list = sorted(get_files(annotations_path, 'txt'))

        for annotation_file in annotations_list:
            # open the annotations file and read it:
            with open(annotation_file) as fptr:
                file_str = fptr.read()
                # separate text into lines and parse each line
                text_list = file_str.split('\n')
                # drop the last element in the list: ''
                text_list = text_list[:-1]

                # skip the file if it has less than two objects:
                if len(text_list) > 1:
                    first_object = text_list[0].split()
                    annotation = {"image_name": first_object[0] + "_" + first_object[1].replace(".fits", ".png"),
                                  "object_classes": list(),
                                  "object_boxes": list()}
                    for obj in text_list:
                        object_fields = obj.split()
                        annotation["object_classes"].append(int(object_fields[2]))
                        object_box = [float(object_fields[3]),
                                      float(object_fields[4]),
                                      float(object_fields[5]),
                                      float(object_fields[6])]
                        annotation["object_boxes"].append(object_box)
                    annotations.append(annotation)
    return annotations


def read_satnet_annotation_file(file):
    with open(file) as f:
        filestr = f.read()
        # separate text into lines and parse each line
        txtlist = filestr.split('\n')
        # drop the last element in the list: '':
        content = txtlist[:-1]
        content = [x.split() for x in content]

        # fix for label: index is off by 1
        # label = [int(x[0])+1 for x in content]
        lbl = [1 for x in content]

        x_center = [float(x[3])/512.0 for x in content]
        y_center = [float(x[4])/512.0 for x in content]
        bbox_width = [float(x[5])/512.0 for x in content]
        bbox_height = [float(x[6])/512.0 for x in content]

        # clip ranges to 0 and 1
        y_min = [max(y0-hig/2, 0.) for y0, hig in zip(y_center, bbox_height)]
        y_max = [min(y0+hig/2, 1.) for y0, hig in zip(y_center, bbox_height)]

        x_min = [max(x0-wid/2, 0.) for x0, wid in zip(x_center, bbox_width)]
        x_max = [min(x0+wid/2, 1.) for x0, wid in zip(x_center, bbox_width)]
        anno = dict()
        anno['class_id'] = lbl

        anno['y_min'] = y_min
        anno['y_max'] = y_max

        anno['x_min'] = x_min
        anno['x_max'] = x_max

        n = len(x_min)

        diff = [0] * n
        trunc = [0] * n
        frontal = 'Frontal'.encode('utf8')
        pose = [frontal] * n

        anno['difficult'] = diff
        anno['truncated'] = trunc
        anno['poses'] = pose
        return anno


class SatnetEvaluationGenerator(EvaluationGenerator):
    def __init__(self, annotations):
        super(SatnetEvaluationGenerator, self).__init__(annotations)

    def _write_tfrecords_file(self, annotations, path_to_tfrecords):
        files_dir = r"\\pds-fs.pacificds.com\Users\gmartin\tensorflow\false_positives\datat\JSON"
        image_str = ""
        annot_str = ""
        actual_imagename_list = list()
        for annotation in annotations:
            actual_imagename_list.append(annotation['image_name'])
            directory, image_name = annotation['image_name'].split('_', 1)
            image_str += os.path.join(files_dir,
                                      directory,
                                      "ImageFiles",
                                      image_name.replace('.png',
                                                         '.jpg')) + '\n'
            annot_str += os.path.join(files_dir,
                                      directory,
                                      "Annotations",
                                      image_name.replace('.png',
                                                         '.txt')) + '\n'

        image_path_list = image_str.split('\n')
        # remove the last '' element from the list
        image_path_list = image_path_list[:-1]

        annot_path_list = annot_str.split('\n')
        # remove the last '' element from the list
        annot_path_list = annot_path_list[:-1]

        num_examples = len(image_path_list)

        # Build a writer for the tfrecord.
        print('Writing to', path_to_tfrecords, '... Num_records = ',
              num_examples)
        writer = tf.python_io.TFRecordWriter(path_to_tfrecords)

        # Iterate over the examples.
        for index in tqdm(range(num_examples)):
            with tf.gfile.GFile(image_path_list[index], 'rb') as fid:
                encoded_jpg = fid.read()
            encoded_jpg_io = io.BytesIO(encoded_jpg)
            image = Image.open(encoded_jpg_io)
            if image.format != 'JPEG':
                raise ValueError('Image format not JPEG')
            key = hashlib.sha256(encoded_jpg).hexdigest()
            # convert to a numpy array:
            image = np.array(image)
            h = int(image.shape[0])
            w = int(image.shape[1])
            c = int(image.shape[2])
            # check if RGBA format and convert to RGB if true:
            if c == 4:
                raise ValueError('Image is RGBA')

            annotations = read_satnet_annotation_file(annot_path_list[index])
            # Note: annotation data is already normalized:
            ymin_norm = annotations['y_min']
            ymax_norm = annotations['y_max']

            xmin_norm = annotations['x_min']
            xmax_norm = annotations['x_max']

            # filename = image_path_list[index].split('/')[-1]
            filename = actual_imagename_list[index]

            label = annotations['class_id']
            text = [CLASS_TEXT[x].encode('utf8') for x in label]
            difficult_obj = annotations['difficult']
            truncated = annotations['truncated']
            poses = annotations['poses']

            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(h),
                'image/width': dataset_util.int64_feature(w),
                'image/filename': dataset_util.bytes_feature(
                    filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(
                    filename.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(
                    key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature(
                    'jpeg'.encode('utf8')),
                'image/object/bbox/ymin': dataset_util.float_list_feature(
                    ymin_norm),
                'image/object/bbox/ymax': dataset_util.float_list_feature(
                    ymax_norm),
                'image/object/bbox/xmin': dataset_util.float_list_feature(
                    xmin_norm),
                'image/object/bbox/xmax': dataset_util.float_list_feature(
                    xmax_norm),
                'image/object/class/text': dataset_util.bytes_list_feature(
                    text),
                'image/object/class/label': dataset_util.int64_list_feature(
                    label),
                'image/object/difficult': dataset_util.int64_list_feature(
                    difficult_obj),
                'image/object/truncated': dataset_util.int64_list_feature(
                    truncated),
                'image/object/view': dataset_util.bytes_list_feature(poses),
            }))
            writer.write(example.SerializeToString())
        writer.close()

    def _infer_and_pickle(self, tfrecords_file_path, output_dir):
        """Run data through model and generate pickled evaluation data

        Args:
            tfrecords_file_path:Path to tfrecords file to evaluate
            output_dir:Directory to store output data and pickle file

        Outputs:
            pickle file with evaluation data in form:
                {"image_name":['unique_img_name', 'unique_img_name'],
                 "detection_scores":[[score, score],[score, score]],
                 "detection_classes":[[class, class],[class, class]],
                 "detection_boxes":[[box, box],[box, box]],
                 "any other relevant data":etc.
                }
                box should be in form:
                    [ymin, xmin, ymax, xmax]
        """

        frozen_graph_path = "C:/Users/Sean Sullivan/PycharmProjects/DataAnalysis/sat_net/exported_graphs/train-3/frozen_inference_graph.pb"
        labels_map_path = "C:/Users/Sean Sullivan/PycharmProjects/DataAnalysis/sat_net/label_data/astronet_label_map_2.pbtxt"

        # Instantiate a batch evaluator - replace for different analysis.
        evaluator = AstroNetEvaluateBatch(
            path_to_frozen_graph=frozen_graph_path,
            path_to_labels_map=labels_map_path)

        # Core inference and pickle function
        # *replace for different analysis.*
        evaluator.process_tfrecords(tfrecords_file_path,
                                    output_dir,
                                    1,
                                    None,
                                    32,
                                    4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations_directory',
                        type=str,
                        default="C:/Users/Sean Sullivan/Documents/share/datat/JSON",
                        help="Base directory containing annotation folders")
    parser.add_argument('--tfrecords_directory',
                        type=str,
                        default="./tfrecords",
                        help="Directory containing tfrecords to evaluate")
    parser.add_argument('--tfrecords_out_directory',
                        type=str,
                        default="./tfrecords",
                        help="Directory to output tfrecords")
    parser.add_argument('--evaluations_out_directory',
                        type=str,
                        default="./evaluations",
                        help="Directory to output evaluation data")
    parser.add_argument('--generate_tfrecords',
                        action="store_true",
                        help="Generate tfrecords from annotations")
    parser.add_argument('--evaluate_tfrecords',
                        action="store_true",
                        help="Evaluate tfrecords")
    parser.add_argument('--num_records_gen',
                        type=int,
                        default=None,
                        help="Max number of tfrecords to split data into")
    parser.add_argument('--num_records_eval',
                        type=int,
                        default=None,
                        help="Max number of records to evaluate")

    args = parser.parse_args()

    _annotations = load_annotations(args.annotations_directory)

    generator = SatnetEvaluationGenerator(_annotations)

    if args.generate_tfrecords:

        generator.generate_tfrecords(args.tfrecords_out_directory,
                                     args.num_records_gen)

    if args.evaluate_tfrecords:

        generator.evaluate_model_on_dataset(args.tfrecords_directory,
                                            args.evaluations_out_directory,
                                            args.num_records_eval)

