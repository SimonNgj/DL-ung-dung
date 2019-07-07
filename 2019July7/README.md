
# Human Activity Recognition on the Wireless Sensor Data Mining (WISDM) dataset using LSTM Recurrent Neural Networks  
This folder contains code used to recognize human activity based on the Wireless Sensor Data Mining (WISDM) dataset using CNN, LSTM and CNN-LSTM.

## Dataset
The data used for classification is provided by the Wireless Sensor Data Mining (WISDM) Lab and can be downloaded  [here](http://www.cis.fordham.edu/wisdm/dataset.php).
It consists of _1,098,207_ examples of various physical activities (sampled at _20Hz_) with _6_ attributes:
`user,activity,timestamp,x-acceleration,y-acceleration,z-acceleration`, and the _activities_ include: `Walking, Jogging, Upstairs, Downstairs, Sitting, Standing`. 

Original research done on this dataset can be found [here](http://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf).


##  Data preprocessing

In order to feed the network with such temporal dependencies a _sliding time window_ is used to extract separate data segments. The _window width_ and the _step size_ can be both adjusted and optimised for better accuracy. Each time step is associated with an activity label, so for each _segment_ the most frequently appearing label is chosen. Here, the _time segment_ or _window width_ is chosen to be _200_ and _time step_ is chosen to be _100_.

### Input:
- data _(data/WISDM_ar_v1.1_raw.txt)_

The data needs to be separated into features and labels and then further into training and test sets. Labels need to be _one-hot_ encoded before feeding into the classifier.

### Output:
- Trained classifier
- Confusion matrix graph
- Error/Accuracy rate graph

## CNN

## LSTM

## CNN-LSTM

## Results
The classifier achieves the accuracy of _96.95%_ by CNN, _92.81%_ by LSTM and _89.85%_ by CNN-LSTM though it might presumably be slightly improved by decreasing the _step size_ of _sliding window_.
The following graphs show the train/test error/accuracy for each epoch and the final confusion matrix (normalised so that each row sums to one).


### Use
1. Run the script with  `python simon.py`
