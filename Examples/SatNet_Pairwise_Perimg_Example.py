from DataAnalyzer import DataAnalyzer, _get_files
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def get_satnet_annotations(root_directory):
    """Retrieve annotations from files and convert to list of dicts

    Args:
        root_directory:Root directory containing annotation directories

    Returns:
        list of dictionaries containing annotations
    """

    print("Retrieving annotations:")

    annotations = list()

    directories = sorted([f for f in os.listdir(root_directory)
                          if os.path.isdir(os.path.join(root_directory, f))])

    for f in tqdm(directories):

        annotations_path = os.path.join(root_directory, f, 'Annotations')

        # get the list of annotation files in the Annotations directory:
        annotations_list = sorted(_get_files(annotations_path, '.txt'))

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


def calc_distance(object1, object2):
    """Calculate distance between two objects

    Args:
        object1:First object in pair, [x, y, w, h]
        object2:Second object in pair, [x, y, w, h]

    Returns:
        float:Distance between object1 and object2
        or
        None if distance greater than 300 or object1/object2 is out of bounds
    """
    x_obj1, y_obj1, _, _ = object1
    x_obj2, y_obj2, _, _ = object2

    # Make sure boxes are in bounds of img, image size: 512x512
    if x_obj1 <= 512 and y_obj1 <= 512 and x_obj2 <= 512 and y_obj2 <= 512:

        distance = np.sqrt(np.square(x_obj2 - x_obj1)
                           + np.square(y_obj2 - y_obj1))

        # Do not include pairs w/ distance > 300
        if distance <= 300:

            return distance

        else:

            return None
    else:

        return None


def least_distance_only(annotation, new_annotations):
    """Condition function to keep only smallest distance annotation per image

    Args:
        annotation: Current annotation being examined
        new_annotations: Dict of new annotations kept by image_name

    Returns: True if annotation image is not yet in new_annotations or distance
             is less than previous value. False otherwise.
    """

    image_name = annotation["image_name"]
    if image_name in new_annotations:

        return annotation["distance"] < new_annotations[image_name]["distance"]
    else:

        return True


if __name__ == "__main__":

    # Specify a root directory for SatNet annotations
    _root_dir = "path/to/annotations/root"

    # Map SatNet annotation text file structure to python list of dicts.
    _annotations = get_satnet_annotations(_root_dir)

    # Instantiate DataAnalyzer, given a list of annotation dicts.
    analyzer = DataAnalyzer(_annotations)

    # Ingest pickle files containing evaluation data.
    analyzer.ingest_multiple_evaluations(directory="path/to/evaluations",
                                         recursive=True)

    # Remove boxes, scores, and classes for evaluations with scores < 0.5.
    analyzer.remove_low_conf_evaluations()

    # Add to analyzer a new attribute containing a list of dicts with computed
    # pairwise features.
    analyzer.new_pairwise_feature(new_feature_name="distance",
                                  feature_extraction_fn=calc_distance,
                                  operand_annotation_key="object_boxes")

    # Calculate TP/FP/FN on pairwise feature evaluations.
    analyzer.calculate_perimg_stats(iou_threshold=.85,
                                    normalized_coordinates=False,
                                    image_size=[512.0, 512.0],
                                    pairwise_annotations=True,
                                    feature="distance")

    # Create plot and axis
    fig, ax1 = plt.subplots()

    # Make a twin axis
    ax2 = ax1.twinx()

    # Make list of distances from pairwise annotations list.
    _sat_distances = [a['distance']
                      for a in analyzer.pairwise_annotations["distance"]]

    # convert distances list to numpy array for matplotlib hist function
    _sat_distances_np = np.array(_sat_distances, dtype=np.float64)

    # plot histogram onto ax1 to show distribution of distances between sats.
    _, _bins, _ = ax1.hist(x=_sat_distances_np,
                           color="blue",
                           alpha=0.5,
                           bins=100)

    # Remove all but smallest distance pair annotation per image
    analyzer.pairwise_to_perimg(feature="distance",
                                condition_fn=least_distance_only)

    # split annotations into bins based on distance between sats
    analyzer.split_annotations(feature="distance",
                               bins=_bins,
                               feature_type=analyzer.SplitTypes.SCALAR,
                               pairwise=True)

    # label and format axis and graph
    ax1.set_xlabel("distance between objects")
    ax1.set_ylabel("# samples")
    ax1.set_title("Distribution of distances between objects")
    ax2.set_ylabel("Precision/Recall of model", color="red")
    ax2.tick_params('y', colors="red")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks(np.arange(0.0, 1.0, 0.1))

    # Calculate the recalls of binned distance data, smooth to 3 bins.
    _recall_bins, _recall_values = analyzer.pairwise_recalls(feature="distance",
                                                             per_img=True,
                                                             smoothing=3)

    # Calculate the precisions of binned distance data, smooth to 3 bins.
    _precision_bins, _precision_values = analyzer.pairwise_precisions(feature="distance",
                                                                      per_img=True,
                                                                      smoothing=3)

    # Plot precision scores over histogram on ax2
    ax2.plot(_precision_bins,
             _precision_values,
             color="red",
             label="Precision")

    # Plot precision scores over histogram on ax2
    ax2.plot(_recall_bins,
             _recall_values,
             color="green",
             label="Recall")

    # Show legend on graph
    plt.legend()

    # Show the graph.
    plt.show()
