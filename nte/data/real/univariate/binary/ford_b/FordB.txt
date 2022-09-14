Problems:

From SDM 2012

This data was originally used in a competition in the IEEE World
Congress on Computational Intelligence, 2008. The classification
problem is to diagnose whether a certain symptom exists or does not
exist in an automotive subsystem. Each case consists of 500
measurements of engine noise and a classification. There are two
separate problems:

For FordB the training data were collected in typical operating
conditions, but the test data samples were collected under noisy
conditions.\\

This distinction with FordB makes it important to maintain the test
and train sets separately (rather than combining and cross
validating). Further details and the competition results can be
found at~\cite{fordCompWebsite}. Some example series are shown in
Figure~\ref{fordExamples}. These graphs indicate that there is
little temporal correlation within each class, and hence that
Euclidean distance based classifiers may perform poorly.

%The winning entries achieved 100\% test accuracy with Ford A and
86.2\% on Ford B. The best results for Ford B were obtained by
constructing a classifiers on the ACF features and/or the Power
Spectrum. The data has not been normalised.
%http://home.comcast.net/~nn_classification/}
