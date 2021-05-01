from layers.conv2d import Conv2d
import textwrap
from typing import Callable, List

import numpy as np
from utils import np2carray
from layers.layer import Layer
from layers.id_gen import IdGenerator
from simulation import conv2d_dw

class Conv2d_DW(Conv2d):
  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    assert self.filter_in_channels == 1 # depthwise means that each filter only acts on one input channel
    input_variable = generate(self.input)

    output_id = id_gen.get_id(self)
    weights_data_count_first = self.filter_data.transpose((3, 0, 1, 2)).reshape((self.filter_count, self.filter_height * self.filter_width))

    data_buffer.append(textwrap.dedent(f"""\
      const elem_t output_{output_id}_weight[{self.filter_count}][{self.filter_height * self.filter_width}] = {np2carray(weights_data_count_first)};
      const acc_t output_{output_id}_bias[{len(self.bias_data)}] = {np2carray(self.bias_data)};
      const float output_{output_id}_multipliers[{self.filter_count}] = {np2carray(self.multipliers)};
      elem_t output_{output_id}[{self.output_height} * {self.output_width}][{self.filter_count}];
    """))

    runtime_buffer.append(textwrap.dedent(f"""\
      conv_auto_dw_multiscale(
        1, {self.input_width}, {self.filter_count},
        {self.filter_count}, {self.output_width},
        {self.stride}, {self.padding}, {self.filter_width},

        {input_variable},  output_{output_id}_weight, output_{output_id}_bias, output_{output_id},

        {"RELU" if self.followed_by_relu else "NO_ACTIVATION"}, output_{output_id}_multipliers, 0,
        1, 0, 0,
        WS
      );"""
    ))
    return f"output_{output_id}"

  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    input_variable = run(self.input)

    # TODO store input and output zero point for each variable
    output = conv2d_dw(input_variable, self.filter_data, self.bias_data, self.multipliers, stride=self.stride)
    return output