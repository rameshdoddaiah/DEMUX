from nte.data.synth.blipv3.blipv3_dataset import BlipV3Dataset

import sys
import json
import torch

pipeline_config={
    "exp_base_name":"pipeline_exp",
    "dataset":BlipV3Dataset(),


}


if __name__ == '__main__':
    args = sys.argv
    pipeline_config = json.load(args[1])

    # Generate dataset /Load Data
    dataset = BlipV3Dataset()

    # Train Black Box Model
    # trained_model = train_black_box_model(data=dataset.test_data, hyper_params=pipeline_params['hyper_params'])
    model = torch.load(pipeline_params['black_box_model_path'])
    model.eval()

    # Generate Saliencies for candidates


    # Generate Saliency Metric
