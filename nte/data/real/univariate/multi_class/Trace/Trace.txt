This 4-class dataset is a subset of the Transient Classification
Benchmark (trace project), an initiative at the turn of the century
to collate data from the application domain of the process industry
(e.g. nuclear, chemical, etc.). It is a synthetic dataset designed
to simulate instrumentation failures in a nuclear power plant,
created by Davide Roverso. The full dataset consists of 16 classes,
50 instances in each class. Each instance has 4 features.

The TRACE subset only uses the second feature of class 2 and the
third feature of class 3 and 7. Hence, this dataset contains 200
instances, 50 for each class. All instances are linearly
interpolated to have the same length of 275 data points, and are
z-normalized.



Classification Error Rates (%):

Euclidean: 11.00%

DTW with 10% warping window size: 0.00%

DTW with the best (3.375%) uniform warping window size: 0.00%
