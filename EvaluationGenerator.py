import os
from abc import abstractmethod


def get_files(directory, extension):
    file_list = []
    for main_dir, _, _ in os.walk(directory):
        for list_files in os.listdir(main_dir):
            if list_files.endswith(extension):
                file_list.append(os.path.normpath(os.path.join(main_dir,
                                                               list_files)))
        return file_list


class EvaluationGenerator:
    def __init__(self, annotations):
        self.annotations = annotations

    def generate_tfrecords(self,
                           output_dir="./tfrecords",
                           num_records=1):
        """Generate tfrecords from annotations

        Args:
            output_dir: Directory to save tfrecords
            num_records: Number of tfrecords files to split data into
                (to make it easier to run evaluations on multiple machines)
        """

        if not os.path.exists(output_dir):

            os.mkdir(output_dir)

        if num_records > 1:

            annotations_list_split = [self.annotations[i::num_records]
                                      for i in range(num_records)]

        else:

            annotations_list_split = self.annotations

        print("Generating tfrecords")

        for i in range(num_records):

            self._write_tfrecords_file(annotations_list_split[i],
                                       os.path.join(output_dir,
                                                    "detections{}.tfrecords".format(i + 1)))

    def evaluate_model_on_dataset(self,
                                  tfrecords_directory='./tfrecords',
                                  output_directory='./evaluations',
                                  number_records=None):
        """Generate evaluations by feeding tfrecords files through model

        Args:
            tfrecords_directory: Directory containing tfrecords file(s)
            output_directory: Directory to output evaluation data
            number_records: Number of records to evaluate, None=All
        """
        print("Evaluating tfrecords")

        # Get all tfrecords files in specified directory
        tfrecords_list = get_files(tfrecords_directory, ".tfrecords")

        # Create the directory, if it does not exists.
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # Run infer_and_pickle_fn on each tfrecords file in specified directory
        for i in range(len(tfrecords_list)):

            # If number of records is specified and reached, stop evaluating.
            if number_records and i >= number_records:

                break

            if not os.path.exists(os.path.join(output_directory, str(i))):

                os.mkdir(os.path.join(output_directory, str(i)))

            self._infer_and_pickle(tfrecords_list[i],
                                   os.path.join(output_directory, str(i)))

    @abstractmethod
    def _write_tfrecords_file(self, annotations, path_to_tfrecords):
        """Write single tfrecords file from annotations
        **Project specific function**
        Args:
            annotations:Annotations to convert into tfrecords file
            path_to_tfrecords:Path to create tfrecords file

        Outputs:
            tfrecords file with converted annotations data
        """

    @abstractmethod
    def _infer_and_pickle(self, tfrecords_file_path, output_dir):
        """Run data through model and generate pickled evaluation data
        **Project specific function**
        Args:
            tfrecords_file_path:Path to tfrecords file to evaluate
            output_dir:Directory to store output data and pickle file

        Outputs:
            pickle file with evaluation data in format:
                {"image_name":['unique_img_name','unique_img_name'],
                 "detection_scores":[[score,score],[score,score]],
                 "detection_classes":[[class,class],[class,class]],
                 "detection_boxes":[[box,box],[box,box]],
                 "any other relevant data":etc.
                }
                box should be in format:
                    [ymin, xmin, ymax, xmax]
        """
