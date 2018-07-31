from itertools import combinations
import pickle
import os
import numpy as np
import operator
import copy


def _get_files(directory=".",
               extension="",
               recursive=False):
    """Get all files with specified extension

    Args:
        directory:Root directory to search for files in
        extension:Extension of files to return
        recursive:Include files found in subdirectories

    Returns:
        list of file paths
    """

    file_list = []

    for main_dir, sub_directories, _ in os.walk(directory):

        for file in os.listdir(main_dir):

            if file.endswith(extension):

                file_list.append(os.path.normpath(os.path.join(main_dir,
                                                               file)))
        if not recursive:

            return file_list

    return file_list


def _iou(pred_box=None,
         gt_box=None):
    """Calculate IoU of single predicted and ground truth box

    Args:
        pred_box (list of floats): location of predicted object as
            [ymin, xmin, ymax, xmax]
        gt_box (list of floats): location of ground truth object as
            [ymin, xmin, ymax, xmax]

    Returns:
        float: value of the IoU for the two boxes.

    Raises:
        AssertionError: if the box is obviously malformed
    """

    if not pred_box or not gt_box:

        print("pred_box and gt_box must be specified")
        raise ValueError

    y1_t, x1_t, y2_t, x2_t = gt_box
    y1_p, x1_p, y2_p, x2_p = pred_box

    if x1_p > x2_p or y1_p > y2_p:

        raise AssertionError(
            "Prediction box is malformed? predicted box: {}\n"
            "Are coords normalized?".format(pred_box))

    if x1_t > x2_t or y1_t > y2_t:

        raise AssertionError(
            "Ground Truth box is malformed? true box: {}\n"
            "Are coords normalized?".format(gt_box))

    if x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t:

        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])

    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)

    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    ioveru = inter_area / (true_box_area + pred_box_area - inter_area)

    return ioveru


class DataAnalyzer:
    class SplitTypes:
        SCALAR = "scalar"
        CLASS = "class"

    def __init__(self, annotations):
        self.annotations = annotations
        self.annotations_binned = dict()
        self.pairwise_annotations = dict()
        self.pairwise_annotations_binned = dict()
        self.evaluations = dict()

    def new_feature(self,
                    new_feature_name="",
                    feature_extraction_fn=None):
        """Generate a new feature from annotations.

        New feature will be added to the annotation dictionary with
        key=new_feature_name

        Args:
            new_feature_name: Name of the new feature to create
            feature_extraction_fn: Function to generate the new feature
                                         value for each annotation
                                         (params=annotations,
                                          return=new feature value)
        """

        if feature_extraction_fn is None:

            print("You must provide a feature_extraction_fn")
            raise ValueError

        for annotation in self.annotations:

            new_feature_value = feature_extraction_fn(annotation)

            if new_feature_value is not None:

                annotation[new_feature_name] = new_feature_value

    def new_pairwise_feature(self,
                             new_feature_name="",
                             feature_extraction_fn=None,
                             operand_annotation_key=""):
        """Generate a new pairwise feature from annotations

        New annotations will be made in:
        self.pairwise_annotations[new_feature_name]

        Args:
            new_feature_name: Name of the new feature to create.

            feature_extraction_fn: Function to generate new pairwise
                feature values for each pair new_feature_function should take
                two objects as input and return the new feature value for that
                pair. Return None to not include the annotation in new
                annotations.

            operand_annotation_key: Feature to generate the pairs from.
                Elements from pairing_feature are referred to as objects
        Return:
            The new annotations with the pairwise feature having
            key: new_feature_name

        object pairs can be found in:
            self.pairwise_annotations[new_feature_name]["pair"]
        in the form of a tuple:
            (object1, object2)
        """

        if feature_extraction_fn is None:
            print("You must provide a feature_extraction_fn")
            raise ValueError

        # Create a list to store the annotation dicts with added features.
        new_annotations = list()

        # Iterate over each annotation dict in the annotations dict list.
        for annotation in self.annotations:

            # Check that there are at least 2 objects for pairwise computations
            if len(annotation[operand_annotation_key]) >= 2:

                # Get a list of 2-combinations of the annotations corresponding
                # to the given key.
                pairs = combinations(annotation[operand_annotation_key], 2)

                # Iterate over each generated 2-combinations.
                for object1, object2 in pairs:

                    # For this 2-combination, extract the relevant feature.
                    new_feature_val = feature_extraction_fn(object1,
                                                            object2)

                    # Ensure that the returned feature value is non-None.
                    if new_feature_val is not None:

                        # Deepcopy the annotation dict to avoid overwriting.
                        new_pair_annotation = copy.deepcopy(annotation)

                        # Add a new dict entry corresponding to
                        # the computed feature.
                        new_pair_annotation[new_feature_name] = new_feature_val

                        # Add a new dict entry for the tuple pair of objects
                        # from which the new feature was computed.
                        new_pair_annotation["pair"] = (object1, object2)

                        # Add the new dict to the list.
                        new_annotations.append(new_pair_annotation)

        # Add a dict entry containing a list of annotation dicts.
        self.pairwise_annotations[new_feature_name] = new_annotations

        return self.pairwise_annotations[new_feature_name]

    def split_annotations(self,
                          feature="",
                          bins=None,
                          feature_type=SplitTypes.SCALAR,
                          pairwise=False):
        """Split annotations into bins based on specified feature

        Args:
            feature (string): Feature to split annotations based on
            bins (list of Scalars or Classes): Bin values to split the feature.
            feature_type: Type of feature specified in feature param
                (default SCALAR)
                SplitTypes: SCALAR, CLASS
            pairwise: Splitting pairwise annotations or not (default False)

        Split annotations placed in self.pairwise_annotations_binned[feature]
                                    or
                                    self.annotations_binned[feature]
        """

        if bins is None:

            print("You must provide a list of bins to split data into")
            raise ValueError

        if pairwise:

            if feature_type == self.SplitTypes.SCALAR:

                self.pairwise_annotations_binned[feature] = list()

                bin_start = bins[0]

                for bin_end in bins[1:]:

                    annotations = list()

                    for annotation in self.pairwise_annotations[feature]:

                        if bin_start <= annotation[feature] <= bin_end:

                            new_annotation = copy.deepcopy(annotation)

                            new_annotation["bin"] = bin_start

                            annotations.append(new_annotation)

                    bin_start = bin_end

                    self.pairwise_annotations_binned[feature].append(annotations)

            elif feature_type == self.SplitTypes.CLASS:

                self.pairwise_annotations_binned[feature] = list()

                for b in bins:

                    annotations = list()

                    for annotation in self.pairwise_annotations[feature]:

                        if b == annotation[feature]:

                            new_annotation = copy.deepcopy(annotation)

                            new_annotation["bin"] = b

                            annotations.append(new_annotation)

                    self.pairwise_annotations_binned[feature].append(annotations)
        else:

            if feature_type == self.SplitTypes.SCALAR:

                self.annotations_binned[feature] = list()

                bin_start = bins[0]

                for bin_end in bins[1:]:

                    annotations = list()

                    for annotation in self.annotations:

                        if bin_start <= annotation[feature] <= bin_end:

                            new_annotation = copy.deepcopy(annotation)

                            new_annotation["bin"] = bin_start

                            annotations.append(new_annotation)

                    bin_start = bin_end

                    self.annotations_binned[feature].append(annotations)

            elif feature_type == self.SplitTypes.CLASS:

                self.annotations_binned[feature] = list()

                for b in bins:

                    annotations = list()

                    for annotation in self.annotations:

                        if b == annotation[feature]:

                            new_annotation = copy.deepcopy(annotation)

                            new_annotation["bin"] = b

                            annotations.append(new_annotation)

                    self.annotations_binned[feature].append(annotations)

    def ingest_evaluation_file(self,
                               evaluation_file=""):
        """Load evaluation data from pickle file

        Args:
            evaluation_file: Path to evaluation pickle file
        """

        with open(evaluation_file, 'rb') as fp:

            evaluations = pickle.load(fp)

        for index, image_name in enumerate(evaluations['image_name']):

            self.evaluations[image_name] = dict()

            for key, value in evaluations.items():

                if key == "image_name":

                    continue

                self.evaluations[image_name][key] = value[index]

    def ingest_multiple_evaluations(self,
                                    directory="",
                                    recursive=False):
        """Ingest multiple evaluation pickle files

        Args:
            directory:Root directory containing pickle files to ingest
            recursive:Recursively ingest pickle files from all subdirectories
        """

        pickles_list = _get_files(directory, ".pickle", recursive)

        for file in pickles_list:

            self.ingest_evaluation_file(file)

    def remove_low_conf_evaluations(self,
                                    score_threshold=0.5):
        """Remove evaluation boxes that have low confidence scores

        Args:
            score_threshold: Confidence level cut off at
        """

        for image_name, evaluation in self.evaluations.items():

            scores = evaluation['detection_scores']
            boxes = evaluation['detection_boxes']
            classes = evaluation['detection_classes']

            indices = [i for i, v in enumerate(scores) if v >= score_threshold]

            evaluation['detection_scores'] = [scores[i] for i in indices]
            evaluation['detection_boxes'] = [boxes[i] for i in indices]
            evaluation['detection_classes'] = [classes[i] for i in indices]

    def pairwise_to_perimg(self,
                           feature="",
                           condition_fn=None):
        """Convert pairwise annotations to have one annotation per img
            based on condition

        Args:
            feature: Pairwise feature annotations to filter
            condition_fn: Function that returns boolean of whether to keep an
                          annotation or not. params given:
                          annotation = the current annotation being examined
                          new_annotations = list of annotations kept so far
        """

        if condition_fn is None:

            print("You must provide a condition_fn")
            raise ValueError

        new_annotations = dict()
        for annotation in self.pairwise_annotations[feature]:

            if condition_fn(annotation, new_annotations):

                new_annotations[annotation["image_name"]] = annotation

        new_annotations_list = list()
        for _, annotation in new_annotations.items():
            new_annotations_list.append(annotation)

        self.pairwise_annotations[feature] = new_annotations_list

    def calculate_pairwise_stats(self,
                                 feature="",
                                 iou_threshold=0.5,
                                 normalized_coordinates=False,
                                 image_size=None):
        """Calculate the TP/FP/FN totals for each feature pairwise annotation

        Args:
            feature: Pairwise feature to calculate totals on
            iou_threshold: Accuracy threshold of boxes to count as correct
            normalized_coordinates: If coordinates are in normalized format:
                                    [ymin, xmin, ymax, xmax]
                                    or coordinate format (default):
                                    [x, y, w, h]
            image_size: Size of annotation images in form [width, height]
                *Required if normalized_coordinates is False
        """

        if not normalized_coordinates and image_size is None:

            print("image_size must be provided when "
                  "normalized_coordinates is False")
            raise ValueError

        for annotation in self.pairwise_annotations[feature]:

            false_positives = list()
            false_negatives = list()
            true_positives = list()

            object1, object2 = annotation["pair"]

            evaluation = self.evaluations[annotation["image_name"]]

            eval_boxes = evaluation["detection_boxes"]

            if not normalized_coordinates:
                # Expected object format: [x,y,w,h]
                xmin = max((float(object1[0]) / image_size[0])
                           - (float(object1[2]) / image_size[0]) / 2, 0.)

                xmax = min((float(object1[0]) / image_size[0])
                           + (float(object1[2]) / image_size[0]) / 2, 1.)

                ymin = max((float(object1[1]) / image_size[1])
                           - (float(object1[3]) / image_size[1]) / 2, 0.)

                ymax = min((float(object1[1]) / image_size[1])
                           + (float(object1[3]) / image_size[1]) / 2, 1.)

                object1 = [ymin, xmin, ymax, xmax]

                xmin = max((float(object2[0]) / image_size[0])
                           - (float(object2[2]) / image_size[0]) / 2, 0.)

                xmax = min((float(object2[0]) / image_size[0])
                           + (float(object2[2]) / image_size[0]) / 2, 1.)

                ymin = max((float(object2[1]) / image_size[1])
                           - (float(object2[3]) / image_size[1]) / 2, 0.)

                ymax = min((float(object2[1]) / image_size[1])
                           + (float(object2[3]) / image_size[1]) / 2, 1.)

                object2 = [ymin, xmin, ymax, xmax]

            compatible_boxes_1 = dict()
            compatible_boxes_2 = dict()
            good_pair_boxes = dict()
            bad_pair_boxes = list()

            # Find all boxes that have iou > threshold and add them to
            # compatible_boxes dicts
            for box_index, box in enumerate(eval_boxes):

                iou_box_1 = _iou(box, object1)
                iou_box_2 = _iou(box, object2)

                if iou_box_1 < iou_threshold > iou_box_2:

                    bad_pair_boxes.append(box)

                elif iou_box_1 >= iou_threshold <= iou_box_2:

                    compatible_boxes_1[box_index] = iou_box_1
                    compatible_boxes_2[box_index] = iou_box_2

                    if iou_box_1 > iou_box_2:

                        pair_list = [object1, object2]

                    else:

                        pair_list = [object2, object1]

                    good_pair_boxes[box_index] = {"pair_list": pair_list,
                                                  "avail_pairs": [],
                                                  "pred": box,
                                                  "matched": False}
                elif iou_box_1 >= iou_threshold:

                    compatible_boxes_1[box_index] = iou_box_1

                    good_pair_boxes[box_index] = {"pair_list": [object1],
                                                  "avail_pairs": [],
                                                  "pred": box,
                                                  "matched": False}
                else:

                    compatible_boxes_2[box_index] = iou_box_2

                    good_pair_boxes[box_index] = {"pair_list": [object2],
                                                  "avail_pairs": [],
                                                  "pred": box,
                                                  "matched": False}

            compatible_boxes_1 = sorted(compatible_boxes_1.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)

            compatible_boxes_2 = sorted(compatible_boxes_2.items(),
                                        key=operator.itemgetter(1),
                                        reverse=True)

            pairing1 = None
            pairing2 = None

            available_objects = [object1, object2]

            r = 0  # Proposal round counter

            # Find stable match for each available object
            while len(available_objects) > 0:

                # Propose objects to compatible boxes in order of preference
                for obj in available_objects:

                    if obj == object1:
                        if len(compatible_boxes_1) < r + 1:

                            available_objects.remove(object1)
                            pairing1 = None
                        else:

                            good_pair_boxes[compatible_boxes_1[r][0]]["avail_pairs"].append(object1)

                    else:

                        if len(compatible_boxes_2) < r + 1:

                            available_objects.remove(object2)
                            pairing2 = None
                        else:

                            good_pair_boxes[compatible_boxes_2[r][0]]["avail_pairs"].append(object2)

                # Let each compatible box pick their preferred box from the
                # ones available to them
                for k, good_box in good_pair_boxes.items():

                    if not good_box["matched"]:

                        if len(good_box["avail_pairs"]) > 1:

                            if good_box["pair_list"][0] == object1:

                                available_objects.remove(object1)
                                pairing1 = (object1, good_box["pred"])
                            else:

                                available_objects.remove(object2)
                                pairing2 = (object2, good_box["pred"])

                            good_box["matched"] = True

                        elif len(good_box["avail_pairs"]) == 1:

                            available_objects.remove(good_box["avail_pairs"][0])

                            if good_box["avail_pairs"][0] == object1:

                                pairing1 = (object1, good_box["pred"])
                            else:

                                pairing2 = (object2, good_box["pred"])

                            good_box["matched"] = True

                        good_box["avail_pairs"] = []

                r += 1

            # Add true positives and false negatives
            if pairing1 is None:

                false_negatives.append(object1)
            else:

                true_positives.append(pairing1)

            if pairing2 is None:

                false_negatives.append(object2)
            else:

                true_positives.append(pairing2)

            # Add false positives
            for k, good_box in good_pair_boxes.items():

                if not good_box["matched"]:

                    false_positives.append(good_box["pred"])

            annotation["pairwise_tp"] = true_positives
            annotation["pairwise_fn"] = false_negatives
            annotation["pairwise_fp"] = false_positives

    def pairwise_recalls(self,
                         feature="",
                         per_img=False,
                         binned=True,
                         smoothing=1):
        """Calculate recalls on specified pairwise dataset

        Args:
            feature: Feature to calc recalls on
            per_img: Calculate per_img recalls or per pair (per pair default)
            binned: Binned data or regular data (default binned)
            smoothing: Number of bins to combine recall values for smoothing
                *Only used if binned is True

        Returns:
            if binned:
                list of bins, list of recalls (elements in lists correspond)
            else:
                list of recalls (corresponds with list of pairwise annotations)
        """

        if binned:

            binned_recalls = list()

            for annotation_list in self.pairwise_annotations_binned[feature]:

                true_positives = 0
                false_negatives = 0

                for annotation in annotation_list:
                    if per_img:

                        true_positives += len(annotation["perimg_tp"])
                        false_negatives += len(annotation["perimg_fn"])

                    else:

                        true_positives += len(annotation["pairwise_tp"])
                        false_negatives += len(annotation["pairwise_fn"])

                recall = true_positives / (true_positives + false_negatives)

                binned_recalls.append((annotation_list[0]["bin"], recall))

            if smoothing > 1:

                recall_bins, recall_values = zip(*binned_recalls)

                divides = len(recall_bins) % smoothing == 0

                new_bins = [x for x in recall_bins[smoothing // 2::smoothing]]

                if not divides:

                    new_bins.append(recall_bins[-1])

                new_recalls = [np.mean(recall_values[i:i + smoothing])
                               for i in range(0, len(recall_values), smoothing)]

                return new_bins, new_recalls
            else:

                return zip(*binned_recalls)
        else:
            recalls = list()

            for annotation in self.pairwise_annotations[feature]:

                if per_img:

                    true_positives = len(annotation["perimg_tp"])
                    false_negatives = len(annotation["perimg_fn"])

                else:

                    true_positives = len(annotation["pairwise_tp"])
                    false_negatives = len(annotation["pairwise_fn"])

                recall = true_positives / (true_positives + false_negatives)

                recalls.append(recall)

            return recalls

    def pairwise_precisions(self,
                            feature="",
                            per_img=False,
                            binned=True,
                            smoothing=1):
        """Calculate precisions on specified pairwise dataset

        Args:
            feature: Feature to calc precisions on
            per_img: Calculate per_img recalls or per pair (per pair default)
            binned: Binned data or regular data (default binned)
            smoothing: Number of bins to combine precision values for smoothing
                *Only used if binned is True

        Returns:
            if binned:
                list of bins, list of precisions (order of lists correspond)
            else:
                list of precisions (corresponds with list of pw annotations)
        """

        if binned:

            binned_precisions = list()

            for annotation_list in self.pairwise_annotations_binned[feature]:

                true_positives = 0
                false_positives = 0

                for annotation in annotation_list:

                    if per_img:

                        true_positives += len(annotation["perimg_tp"])
                        false_positives += len(annotation["perimg_fp"])

                    else:

                        true_positives += len(annotation["pairwise_tp"])
                        false_positives += len(annotation["pairwise_fp"])

                precision = true_positives / (true_positives + false_positives)

                binned_precisions.append((annotation_list[0]["bin"], precision))

            if smoothing > 1:

                precision_bins, precision_values = zip(*binned_precisions)

                divides = len(precision_bins) % smoothing == 0

                new_bins = [x for x in precision_bins[smoothing//2::smoothing]]

                if not divides:

                    new_bins.append(precision_bins[-1])

                new_precisions = [np.mean(precision_values[i:i + smoothing])
                                  for i in range(0, len(precision_values), smoothing)]

                return new_bins, new_precisions
            else:

                return zip(*binned_precisions)

        else:

            precisions = list()

            for annotation in self.pairwise_annotations[feature]:

                if per_img:

                    true_positives = len(annotation["perimg_tp"])
                    false_positives = len(annotation["perimg_fp"])

                else:

                    true_positives = len(annotation["pairwise_tp"])
                    false_positives = len(annotation["pairwise_fp"])

                precision = true_positives / (true_positives + false_positives)

                precisions.append(precision)

            return precisions

    def calculate_perimg_stats(self,
                               iou_threshold=0.5,
                               normalized_coordinates=False,
                               image_size=None,
                               pairwise_annotations=False,
                               feature=""):
        """Calculate the TP/FP/FN totals for each annotation

        Args:
            iou_threshold: Accuracy threshold of boxes to count as
                           correct
            normalized_coordinates: Coordinates in normalized format:
                                    [ymin, xmin, ymax, xmax] - True
                                    or coordinate format (default):
                                    [x, y, w, h] - False
            image_size: Size of annotation images in form [w, h]
                *Required if normalized_coordinates is False
            pairwise_annotations: If annotations are from pairwise list or not
            feature: Pairwise feature to use annotations from
                *Only needed if pairwise_annotations is True
        """

        if not normalized_coordinates and image_size is None:

            print("image_size must be provided when"
                  " normalized_coordinates is False")
            raise ValueError

        if pairwise_annotations:

            annotations = self.pairwise_annotations[feature]

        else:

            annotations = self.annotations

        for annotation in annotations:

            # Get evaluations data for annotation
            evaluation = self.evaluations[annotation["image_name"]]
            eval_boxes = evaluation["detection_boxes"]

            objects = list()
            compatible_boxes = dict()
            available_objects = list()

            passed_objects = 0

            # Get all objects in annotation
            for obj_index, obj in enumerate(annotation["object_boxes"]):

                if not normalized_coordinates:

                    xmin = max((float(obj[0]) / image_size[0])
                               - (float(obj[2]) / image_size[0]) / 2, 0.)

                    xmax = min((float(obj[0]) / image_size[0])
                               + (float(obj[2]) / image_size[0]) / 2, 1.)

                    ymin = max((float(obj[1]) / image_size[1])
                               - (float(obj[3]) / image_size[1]) / 2, 0.)

                    ymax = min((float(obj[1]) / image_size[1])
                               + (float(obj[3]) / image_size[1]) / 2, 1.)

                    if xmin > 1 or xmax > 1 or ymin > 1 or ymax > 1:

                        passed_objects += 1

                    else:

                        compatible_boxes[obj_index - passed_objects] = list()
                        available_objects.append(obj_index - passed_objects)
                        objects.append([ymin, xmin, ymax, xmax])
                else:

                    compatible_boxes[obj_index] = list()
                    available_objects.append(obj_index)
                    objects.append(obj)

            good_pair_boxes = dict()
            bad_pair_boxes = list()

            # Find all compatible boxes
            for box_index, box in enumerate(eval_boxes):

                pair_box = {"pref_list": list(),
                            "avail_pairs": list(),
                            "box": box,
                            "matched": False}
                good_pair = False

                # Compare every ground truth object with eval box
                for obj_index, obj in enumerate(objects):

                    iou = _iou(box, obj)

                    if iou >= iou_threshold:

                        good_pair = True
                        compatible_boxes[obj_index].append((box_index, iou))
                        pair_box["pref_list"].append((obj_index, iou))

                if good_pair:

                    good_pair_boxes[box_index] = pair_box

                else:

                    bad_pair_boxes.append(box_index)

            # Sort compatible box lists by iou descending
            for key, box_list in compatible_boxes.items():

                compatible_boxes[key] = sorted(box_list,
                                               key=operator.itemgetter(1),
                                               reverse=True)

            pairings = list()
            not_paired = list()
            r = 0  # Proposal round counter

            # Find stable match for each available object
            while len(available_objects) > 0:
                # Propose objects to compatible boxes in order of preference
                for obj_index in available_objects:

                    if len(compatible_boxes[obj_index]) < r + 1:

                        available_objects.remove(obj_index)
                        not_paired.append(obj_index)
                    else:

                        good_pair_boxes[compatible_boxes[obj_index][r][0]]["avail_pairs"].append(obj_index)

                # Let each eval box pick their preferred box
                # from the ones available to them
                for key, good_box in good_pair_boxes.items():

                    if not good_box["matched"]:

                        if len(good_box["avail_pairs"]) >= 1:

                            if len(good_box["avail_pairs"]) == 1:

                                pref_avail = good_box["avail_pairs"]
                            else:

                                pref_list = sorted(good_box["pref_list"],
                                                   key=operator.itemgetter(1),
                                                   reverse=True)

                                pref_list = [x[0] for x in pref_list]

                                pref_avail = sorted(good_box["avail_pairs"],
                                                    key=lambda x: pref_list.index(x))

                            pairings.append((pref_avail[0], key))
                            available_objects.remove(pref_avail[0])

                            good_box["matched"] = True
                r += 1

            false_positives = list()

            # Add up false positives
            for key, good_box in good_pair_boxes.items():
                if not good_box["matched"]:
                    false_positives.append(key)

            # Add TP/FN/FP to annotation
            annotation["perimg_tp"] = pairings
            annotation["perimg_fn"] = not_paired
            annotation["perimg_fp"] = false_positives + bad_pair_boxes
