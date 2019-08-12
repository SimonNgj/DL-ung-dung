
# Human Activity Recognition using DL

Sự phát triển vượt bậc trong công nghệ gần đây đã góp phần thúc đẩy những nghiên cứu trong lĩnh vực nhận dạng hành động con người (human activity recognition - HAR). HAR từ lâu đã được ứng dụng trong nhiều lĩnh vực như tương tác người-máy, quân sự, an ninh và đặc biệt trong các dịch vụ y tế, chăm sóc sức khỏe. Ở những nước tiên tiến, HAR đã được sử dụng để theo dõi hoạt động của người già ở những trung tâm hưu trí hay những cơ sở y tế phục hồi chức năng, góp phần phát hiện sớm những bất thường như đột quỵ, suy giảm chức năng vận động.

This folder contains code used to recognize human activity based on the Wireless Sensor Data Mining (WISDM) dataset using 
- FCN      | MC-FCN
- LSTM     | Bidirectional LSTM
- FCN-LSTM
- Resnet

This post highly relies on this repo: https://github.com/bartkowiaktomasz/har-wisdm-lstm-rnns

## Dataset
The data used for classification is provided by the Wireless Sensor Data Mining (WISDM) Lab and can be downloaded  [here](http://www.cis.fordham.edu/wisdm/dataset.php).
It consists of _1,098,207_ examples of various physical activities (sampled at _20Hz_) with _6_ attributes:
`user,activity,timestamp,x-acceleration,y-acceleration,z-acceleration`, and the _activities_ include: `Walking, Jogging, Upstairs, Downstairs, Sitting, Standing`. 

Original research done on this dataset can be found [here](http://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf).

##  Data preprocessing
In order to feed the network with such temporal dependencies a _sliding time window_ is used to extract separate data segments. The _window width_ and the _step size_ can be both adjusted and optimised for better accuracy. Each time step is associated with an activity label, so for each _segment_ the most frequently appearing label is chosen. Here, the _time segment_ or _window width_ is chosen to be _200_ and _time step_ is chosen to be _100_.

 |60% for training | 20% for validating | 20% for testing|
 |-----------------|--------------------|----------------|

### Input:
- data _(data/WISDM_ar_v1.1_raw.txt)_

The data needs to be separated into features and labels and then further into training and test sets. Labels need to be _one-hot_ encoded before feeding into the classifier.

### Output:
- Trained classifier
- Confusion matrix graph
- Error/Accuracy rate graph

### Parameters
- SEGMENT_TIME_SIZE = 200: Window length
- N_FEATURES = 3: Number of features (3 sensors)
- N_CLASSES = 6: number of activities
- Learning rate = 0.0025
- BATCH_SIZE = 32

## Results
The following graphs show the train/test error/accuracy for each epoch and the final confusion matrix (normalised so that each row sums to one).

| Ref[1] | Ref[2] |   LSTM  |Bi-LSTM | FCN     | MC-FCN  |FCN-LSTM | Resnet  |
|--------|--------|---------|--------|---------|---------|---------|---------|
| 97.63% |  91.70%| 92.81%| 93.13% | 96.95%  | 94.95%  | 89.85%  |  98.82% |

### Use
1. Run the script with  `python simon.py`

### References
1. A. Ignatov, *Real-time human activity recognition from accelerometer data using Convolutional Neural Networks*, Applied Soft Computing, pp. 915-922, 2018.
2. JR. Kwapisz et al, *Activity Recognition using Cell Phone Accelerometers*, ACM SIGKDD, Vol. 12 Issue 2, pp. 74-82, 2011
