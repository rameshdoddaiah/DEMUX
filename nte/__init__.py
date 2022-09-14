import os

NTE_MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
NTE_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))+"/models/"
NTE_TRAINED_MODEL_PATH = os.path.dirname(os.path.abspath(__file__))+"/trained_models/"
NTE_SALIENCY_PATH = os.path.dirname(os.path.abspath(__file__))+"/saliencies/"
METRICS_PATH = os.path.dirname(os.path.abspath(__file__))+'/../benchmarks/'
PARAMS_PATH = os.path.dirname(os.path.abspath(__file__))+'/../params/'
SALIENCY_PATH = os.path.dirname(os.path.abspath(__file__))+'/saliencies/'
RESULTS_PATH = os.path.dirname(os.path.abspath(__file__))+'/../results/pipeline/'

from nte.bbm_model_trainer import fit
