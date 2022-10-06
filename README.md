<!-- Start of Badges -->
![version badge](https://img.shields.io/badge/rainbow--print%20version-0.0.0-green.svg)
![build](https://github.com/kingspp/rainbow-print/workflows/Release/badge.svg)
![coverage badge](https://img.shields.io/badge/coverage-0.00%25|%200.0k/0k%20lines-green.svg)
![test badge](https://img.shields.io/badge/tests-0%20total%7C0%20%E2%9C%93%7C0%20%E2%9C%98-green.svg)
![docs badge](https://img.shields.io/badge/docs-none-green.svg)
![commits badge](https://img.shields.io/badge/commits%20since%20v0.0.0-0-green.svg)
![footprint badge](https://img.shields.io/badge/mem%20footprint%20-0.00%20Mb-green.svg)
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/ramesh.doddaiah@gmail.com)
<!-- End of Badges -->

# DEMUX
#ICDM 2022 research paper code repository , runtime performance and hyper parameter table

https://icdm22.cse.usf.edu/

**Class-Specific Explainability for Deep Time Series Classifiers**

Abstract—Explainability helps users trust deep learning solu-tions for time series classification. However, existing explainability methods for multi-class time series classifiers focus on one class at a time, ignoring relationships between the classes. Instead, when a classifier is choosing between many classes, an effective explanation must show what sets the chosen class apart from the rest. We now formalize this notion, studying the open problem of class-specific explainability for deep time series classifiers, a challenging and impactful problem setting. We design a novel explainability method, DEMUX, which learns saliency maps for explaining deep multi-class time series classifiers by adaptively ensuring that its explanation spotlights the regions in an input time series that a model uses specifically to its predicted class. DEMUX adopts a gradient-based approach composed of three interdependent modules that combine to generate consistent, class-specific saliency maps that remain faithful to the classifier’s behavior yet are easily understood by end users. Our experimental study demonstrates that DEMUX outperforms nine state-of-the-
art alternatives on five popular datasets when explaining two types of deep time series classifiers. Further, through a case study, we demonstrate that DEMUX’s explanations indeed highlight what separates the predicted class from the others in the eyes of the classifier.

**XAI methods run time performance on ACSF1 dataset and Fully Connected Network**


![plot](./Class-Specific%20XAI%20Methods%20Runtime%20Performance.jpg)




**Hyper Parameter Table**


![plot](./Hyper%20parameter%20table.jpg)



## Requirements
Python 3.7+

### Development
```bash
# Bare installation
git clone https://github.com/rameshdoddaiah/DEMUX

# Install requirements
cd DEMUX && pip install -r requirements.txt
```

## Reproduction
```bash
python3 main.py --pname ICDM --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --jobs_per_task 10\
                 --samples_per_task 10\
                 --dataset Trace-mc\
                 --algo lmux_v1\
                 --p_buf_init cluster\
                 --seed_value 1\
                 --enable_lr_decay False \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test 
                 --background_data_perc 100\
                 --enable_seed True \
                 --max_itr 5000\
                 --ssd dynamic\
                 --run_id 1 \
                 --bbm dnn \
                 --eval_replacement_mc cluster_mean\
                 --multi_class yes\
                 --mse yes\
                 --kl yes\
                 --num_perturbs 500\
                 --mask_mse_coeff 1.0\
                 --mse_coeff 1.0\
                 --enable_tvnorm True\
                 --enable_budget True\
                 --dataset_type test\
                 --clusters mean\
                 --num_masks 5000\
                 --l_prev kld\
                 --l_budget_coeff 0.6\
                 --run 1\
                 --l_ssd_coeff 0.01\
                 --l_tv_norm_coeff 0.5\
                 --iou_alpha 0.5\
                 --l_prev_coeff 1.0\
                 --l_max_coeff 0.7\
```


## Cite
```bash
@inproceedings{doddaiah2022classspecific,
  author  = {Doddaiah, Ramesh and Parvatharaju, Prathyush and Hartvigsen, Thomas and Rundensteiner, Elke},
  title   = {Class-Specific Explainability for Deep Time Series Classifiers},
  booktitle = {IEEE International Conference on Data Mining},
  year    = 2022,
}
```
