https://wandb.ai/xai/multi_class_cricketx?workspace=user-rameshdoddaiah
run 160  MC: one sample_mask or last good mask
run 157 with mc, new auc = true and without preserve probability
run 161 (without mc)
https://docs.google.com/spreadsheets/d/10EuzMkx0LRLqvHxFLFSzIJyRXD7fcBq0bjv4m8XAXLU/edit#gid=0

										eval_replacement_mc		grad_replacement
157	old auc =no	new auc = yes	preserve prob =no		steps =20	test				min_class_mean		random_instance
158	old auc =no	new auc = yes	preserve prob =yes		steps = 20	test				min_class_mean		random_instance
159	old auc = no	new auc =no	preserve_prob=yes		steps =20	test				min_class_mean		random_instance
160	old auc = no	new auc =no	preserve_prob=no		steps =20	test	sample_masks=1 or various_perturbations[-1]			min_class_mean		random_instance
161	old auc =yes	new auc =no	preserve_prob=no		steps =20	test				min_class_mean		random_instance
