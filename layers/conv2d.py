from dataclasses import dataclass
import textwrap
from typing import Callable, List

import numpy as np

from utils import np2carray
from layers.layer import Layer
from layers.id_gen import IdGenerator
from simulation import conv2d

@dataclass
class Conv2d(Layer):
  input: Layer

  filter_width: int
  filter_height: int
  filter_in_channels: int
  filter_count: int
  filter_data: np.ndarray
  bias_data: np.ndarray

  input_width: int
  input_height: int

  stride: int
  padding: int
  followed_by_relu: bool

  output_scale: float
  filter_scales: List[float]

  output_width: int
  output_height: int

  @property
  def multipliers(self):
    return [self.input.output_scale * filter_scale / self.output_scale for filter_scale in self.filter_scales]
    
  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    input_variable = generate(self.input)

    output_id = id_gen.get_id(self)
    filter_data_orig_input_flattened = self.filter_data.reshape((self.filter_count, self.filter_height * self.filter_width * self.filter_in_channels))

    data_buffer.append(textwrap.dedent(f"""\
      const acc_t output_{output_id}_bias[{len(self.bias_data)}] = {np2carray(self.bias_data)};
      const float output_{output_id}_multipliers[{self.filter_count}] = {np2carray(self.multipliers)};
      const elem_t output_{output_id}_weight_filterFirst[{self.filter_count}][{self.filter_width * self.filter_height * self.filter_in_channels}] = {np2carray(filter_data_orig_input_flattened)};
      elem_t output_{output_id}[{self.output_height} * {self.output_width}][{self.filter_count}];"""
    ))

    runtime_buffer.append(textwrap.dedent(f"""\
      conv_auto_multiscale(
        1, {self.input_width}, {self.filter_in_channels},
        {self.filter_count}, {self.output_width},
        {self.stride}, {self.padding}, {self.filter_width},
        
        {input_variable}, output_{output_id}_weight_filterFirst, output_{output_id}_bias, output_{output_id},

        {"RELU" if self.followed_by_relu else "NO_ACTIVATION"}, output_{output_id}_multipliers, 0,
        1, 1, 0,
        WS
      );"""
    ))
    return f"output_{output_id}"
 
  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    input_variable = run(self.input)

    # TODO store input and output zero point for each variable
    output = conv2d(input_variable, self.filter_data, self.bias_data, self.multipliers, stride=self.stride, padding=self.padding)
    return output