# DEMUX
#ICDM 2022 research paper code repository , runtime performance and hyper parameter table
#https://icdm22.cse.usf.edu/

Class-Specific Explainability for Deep Time Series Classifiers

Abstract—Explainability helps users trust deep learning solu-tions for time series classification. However, existing explainability methods for multi-class time series classifiers focus on one class
at a time, ignoring relationships between the classes. Instead, when a classifier is choosing between many classes, an effective explanation must show what sets the chosen class apart from the
rest. We now formalize this notion, studying the open problem of class-specific explainability for deep time series classifiers, a challenging and impactful problem setting. We design a novel
explainability method, DEMUX, which learns saliency maps for explaining deep multi-class time series classifiers by adaptively ensuring that its explanation spotlights the regions in an input
time series that a model uses specifically to its predicted class. DEMUX adopts a gradient-based approach composed of three interdependent modules that combine to generate consistent,
class-specific saliency maps that remain faithful to the classifier’s behavior yet are easily understood by end users. Our experimental study demonstrates that DEMUX outperforms nine state-of-the-
art alternatives on five popular datasets when explaining two types of deep time series classifiers. Further, through a case study, we demonstrate that DEMUX’s explanations indeed highlight
what separates the predicted class from the others in the eyes of the classifier.

https://github.com/rameshdoddaiah/DEMUX/blob/master/Class-Specific%20XAI%20Methods%20Runtime%20Performance.jpg