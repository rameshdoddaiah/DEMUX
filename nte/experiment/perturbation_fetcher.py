# -*- coding: utf-8 -*-
"""
| **@created on:** 11/22/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""
import wandb

entity = 'xai'
project = 'perturbation_analysis_v1'

api = wandb.Api()

runs = api.runs(f"{entity}/{project}")

for r in runs:
    if 'random' in r.name:
        print(r.name)
        for f in r.files():
            if 'perturbations' in f.name:
                f.download("/Users/prathyushsp/Git/TimeSeriesSaliencyMaps/notebooks/perturbation_analysis/random/")



