import logging
from os.path import join
from typing import Dict
import numpy as np

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args: Dict[str, str]) -> None:
        self.model_path = join(args["model_repository"], args["model_version"], "checkpoints")
        self.model_instance_name = args["model_instance_name"]

    def execute(self, requests):
        responses = []

        logging.info(
            f"{self.model_instance_name}: Got {len(requests)} {'request' if len(requests) == 1 else 'requests'}"
        )

        for request in requests:
            model_softmax_output = pb_utils.get_input_tensor_by_name(request, "fc_1").as_numpy()

            boolean_output = True if model_softmax_output[0][1] > model_softmax_output[0][0] else False
            probability_output = model_softmax_output[0][1]
            
            response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("boolean_output", np.array(boolean_output)),
                pb_utils.Tensor("probability_output", np.array(probability_output))
            ])
            responses.append(response)

        logging.info(
            f"{self.model_instance_name}: Processed {len(requests)} {'request' if len(requests) == 1 else 'requests'}"
        )

        return responses
