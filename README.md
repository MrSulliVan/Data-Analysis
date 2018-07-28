# DataAnalyzer
Module to help analyze machine learning datasets

## Motivations
The DataAnalyzer was created to make it easier to generate meaningful graphs based on key features in object detection datasets.

## Usage
You can use a simple work flow when using the DataAnalyzer:

### Format annotations
Since every object detection model is structured differently, a standardized annotation format was needed.

We decided that each annotation should be in the form of a dictionary.
Only three main keys need to be used when converting your annotations to the standardized form:
* **"image_name"** - A list of **unique** image names(or just unique names) for each annotation.
* **"object_classes"** - A list containing the object class for each ground truth object.
						[class,class] classes can be integers or strings
* **"object_boxes"** - A list containing the object box for each ground truth object.
						[box,box] boxes can be in the form [ymin, xmin, ymax, xmax] or [x, y, width, height].
#### Example
Using SatNet as an example, we store our annotations in text files where each text file contains a line for each ground truth object in the image.
It is in a structure like the following:
```
Source
├───site1
│   ├───Annotations
│   │       sat_1234.0001.txt
│   │       sat_1234.0002.txt
│   │       sat_1234.0003.txt
│   └───ImageFiles
│           sat_1234.0001.jpg
│           sat_1234.0002.jpg
│           sat_1234.0003.jpg
├───site2
    ├───Annotations
    │       sat_1234.0001.txt
    │       sat_1234.0002.txt
    │       sat_1234.0003.txt
    └───ImageFiles
            sat_1234.0001.jpg
            sat_1234.0002.jpg
            sat_1234.0003.jpg
```
So to convert this to the required format, we must first extract all the relevant information from the text file and end up with:
```
directory = "site1"
filename = "sat_1234.0001.txt"
gt1 = x1, y1, w1, h1, class1
gt2 = x2, y2, w2, h2, class2
```
Which we then put into our three required dictionary keys:
```
{"image_name": directory + "_" + filename,
 "object_classes": [class1, class2]
 "object_boxes": [[x1, y1, w1, h1],[x2, y2, w2, h2]]}
```
Note: Since we can have the same filenames in each of our site directories, we need to make the image_name is unique by combining the directory they are in with the filename.
We throw each of these dicts into a list to form our annotations list.

### Instantiate DataAnalyzer object
Now that we have the annotations in the correct format, we can make a new DataAnalyzer object by sending in our formatted annotations:
```python
analyzer = DataAnalyzer(_annotations)
```

### Ingest Evaluation Data
Each of our annotations needs corresponding evaluation data to calculate accuracy statistics.

We get this data by running our annotations through our model and storing the output into a pickle file.
You can create a class which inherits [EvaluationGenerator](EvaluationGenerator.py) in order to generate this evaluation data.

Once the evaluation pickle file is made, it is simple to get it into the DataAnalyzer:

```python
analyzer.ingest_evaluation_file("path/to/file.pickle")
```
or if the evaluations are separated into multiple pickle files:
```python
analyzer.ingest_multiple_evaluations("path/to/directory/containing/pickle/files")
```
to remove detection boxes which have a low score we call:
```python
analyzer.remove_low_conf_evaluations()
```

### Generate new features
Now that we have all the information we need in the DataAnalyzer, we can start creating new features which we can use in various forms of data analysis.

In this case, we want to create a new pairwise feature which will give us the distance between pairs of satellites in our images.

To do this we will use the `new_pairwise_feature` function. This function requires a feature extraction function as input which we created before:
```python
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
```
This function takes the coordinates of the two satellites and outputs the distance between them if the distance is less than 300.

We can now use this function in the `new_pairwise_feature` function to generate our new feature:
```python
analyzer.new_pairwise_feature(new_feature_name="distance",
			     feature_extraction_fn=calc_distance,
			     operand_annotation_key="object_boxes")
```
We now have a new set of annotations which was stored in `analyzer.pairwise_annotations["distance"]`.

### Calculate Stats
Now that we have the feature that we want to evaluate, we can calculate the stats on that feature.

Since we made a pairwise feature we will use `calculate_pairwise_stats`:
```python
analyzer.calculate_pairwise_stats(feature="distance",
				  iou_threshold=.85,
				  normalized_coordinates=False,
				  image_size=[512.0, 512.0])
```
This will store the True Positive, False Positive, and False Negatives as a list for each annotation in the `pairwise_annotations["distance"]` list.
Each annotation dict will have new keys for each:
```python
{"pairwise_tp",
 "pairwise_fp",
 "pairwise_fn"}
```

### Create the graph
Time to get this data onto a graph to visualize!

Creating the graph axes:
```python
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel("distance between objects")
ax1.set_ylabel("# samples")
ax1.set_title("Distribution of distances between objects")
ax2.set_ylabel("F1 of model", color="red")
ax2.tick_params('y', colors="red")
ax2.set_ylim(0.0, 1.0)
ax2.set_yticks(np.arange(0.0, 1.0, 0.1))
```
Put pair distances into a list to use on a histogram:
```python
_sat_distances = [a['distance']
		  for a in analyzer.pairwise_annotations["distance"]]

# convert distances list to numpy array for matplotlib hist function
_sat_distances_np = np.array(_sat_distances, dtype=np.float64)

# plot histogram onto ax1 to show distribution of distances between sats.
_, _bins, _ = ax1.hist(x=_sat_distances_np,
		       color="blue",
		       alpha=0.5,
		       bins=100)
```

### Split the Annotations
We now have split our distances into bins for our histogram, but our annotations still need to be split into these bins to generate per-bin accuracy measures.

We can do this by using the `analyzer.split_annotations` function and giving it the bins we made for our histogram:
```python
# split annotations into bins based on distance between sats
analyzer.split_annotations(feature="distance",
			   bins=_bins,
			   feature_type=analyzer.SplitTypes.SCALAR,
			   pairwise=True)
```
This gives us yet another set of annotations which are split into the given bins.
They are stored in `analyzer.pairwise_annotations_binned["distance"]`.

### Calculate Accuracies
With the split annotations, we can do bin recall and precision measurements:
```python
# Calculate the recalls of binned distance data, smooth to 3 bins.
_recall_bins, _recall_values = analyzer.binned_pairwise_recalls(feature="distance",
								smoothing=3)

# Calculate the precisions of binned distance data, smooth to 3 bins.
_precision_bins, _precision_values = analyzer.binned_pairwise_precisions(feature="distance",
									 smoothing=3)
```
We will then combine these measurements into an F1 score to plot on our graph:
```python
_f1_values = list()

# Calculate F1 scores based on binned recall and precision values.
for _recall, _precision in zip(_recall_values, _precision_values):
	_f1_value = 2 * (_precision * _recall) / (_precision + _recall)
	_f1_values.append(_f1_value)

# Plot F1 scores over histogram on ax2, recall bins are the same as F1 bins
ax2.plot(_recall_bins,
	 _f1_values,
	 color="red")
# Show the graph.
plt.show()
```
Finally we get to see our hard work pay off with a beautiful (and extremely useful) graph:

![alt text](https://github.com/MrSulliVan/Data-Analysis/blob/master/output.png "Output Graph")

## Authors

* **Sean Sullivan** - https://github.com/MrSulliVan/

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
