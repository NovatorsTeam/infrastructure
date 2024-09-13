import logging
from os.path import join
from typing import Dict

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F
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
            bottom_image_tensor = pb_utils.get_input_tensor_by_name(request, "bottom_image")
            side_1_image_tensor = pb_utils.get_input_tensor_by_name(request, "side_1_image")
            side_2_image_tensor = pb_utils.get_input_tensor_by_name(request, "side_2_image")
            side_3_image_tensor = pb_utils.get_input_tensor_by_name(request, "side_3_image")
            side_4_image_tensor = pb_utils.get_input_tensor_by_name(request, "side_4_image")

            bottom_image = F.to_grayscale(torch.from_numpy(bottom_image_tensor.as_numpy()))
            height, width = bottom_image.shape[-2:]

            side_images = [
                torch.from_numpy(side_1_image_tensor.as_numpy()),
                torch.from_numpy(side_2_image_tensor.as_numpy()),
                torch.from_numpy(side_3_image_tensor.as_numpy()),
                torch.from_numpy(side_4_image_tensor.as_numpy())
            ]

            side_images = [F.resize(F.to_grayscale(side_image), (height // 4, width)) for side_image in side_images]
            side_concatination = F.resize(torch.cat(side_images, dim=2), (height, width))
            input_image = torch.cat([bottom_image, side_concatination], dim=1)
            input_image = F.resize(input_image, (320, 320))

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("l_x_", np.array(input_image)),
                ]
            )
            responses.append(response)

        logging.info(
            f"{self.model_instance_name}: Processed {len(requests)} {'request' if len(requests) == 1 else 'requests'}"
        )

        return responses
