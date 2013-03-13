We provide the two datasets (osr and pubfig), the learnt relative attributes and their predictions used in the following paper:

Devi Parikh and Kristen Grauman
Relative Attributes
International Conference on Computer Vision (ICCV), 2011 (Oral).

@InProceedings{relative_attributes,
author = {D. Parikh and K. Grauman},
title = {Relative Attributes},
booktitle = {IEEE International Conference on Computer Vision (ICCV)},
month = {Nov},
year = {2011}
}

If you use any part of this data, please cite the above paper.

Each directory (osr and pubfig) contain the following:

images: The images for the pubfig dataset contain cropped faces re-sized to be 256x256. These faces have been extracted from a subset of the images in the PubFig dataset (http://www.cs.columbia.edu/CAVE/databases/pubfig/) of Kumar et al. The images for the ors dataset are directly taken from the outdoor scene recognition dataset of Oliva and Torralba (http://people.csail.mit.edu/torralba/code/spatialenvelope/). A link to a downloadable folder of images has been provided in images.txt

data.mat: Loading this file in MATLAB creates the following variables in the workspace:

im_names: Names of the images

attribute_names: Names of the attributes

class_names: Names of the categories

class_labels: The ground-truth category label of each image

feat: Features extracted for each image (gist for ors, gist concatenated with color for pubfig)

binary_predicates: A binary matrix indicating the presence of an attribute for each category. This was used as ground truth to train binary attribute predictors.

relative_ordering: The relative ordering of all categories for each of the attributes. Low values indicate a weak relative strength of the attribute in the class. Equal values indicate similar relative strengths of the attribute. Let's consider attribute i and classes j and k. relative_ordering(i,j)>relative(i,k) indicates that class j has a stronger presence of attribute i than class k. This was used as ground-truth to train relative attribute predictors using a learning to rank formulation as described in the paper.

relative_att_predictors: The weight matrix of learnt linear relative attribute predictors.

relative_att_predictions: The score of each image for each relative attribute. relative_att_predictions = feat*relative_att_predictors.

used_for_training: A binary indicator that specifies which of the images were used to train the relative attribute predictors.
