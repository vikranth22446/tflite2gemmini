from dataclasses import dataclass
import textwrap
from typing import Callable, List

import numpy as np

from utils import np2carray
from layers.layer import Layer
from layers.id_gen import IdGenerator
from simulation import fc
@dataclass
class FC(Layer):
  input: Layer
  output_size: int
  input_size: int

  weight_data: np.ndarray
  weight_scale: float

  bias_data: np.ndarray

  output_scale: float
  followed_by_relu: bool
  
  weight_zero_point:int
  output_zero_point:int

  @property
  def multipliers(self):
    return self.input.output_scale * self.weight_scale / self.output_scale
    
  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    input_variable = generate(self.input)

    output_id = id_gen.get_id(self)

    data_buffer.append(textwrap.dedent(f"""\
      const elem_t output_{output_id}_weight[{self.output_size}][{self.input_size}] = {np2carray(self.weight_data)};
      const acc_t output_{output_id}_bias[{len(self.bias_data)}] = {np2carray(self.bias_data)};
      elem_t output_{output_id}[{self.output_size}][1];"""
    ))

    runtime_buffer.append(textwrap.dedent(f"""\
      tiled_matmul_nn_auto(
        {self.output_size}, 1, {self.input_size},
        output_{output_id}_weight, {input_variable}, output_{output_id}_bias, output_{output_id},
        {"RELU" if self.followed_by_relu else "NO_ACTIVATION"}, {self.multipliers}, 0, false,
        WS, false, "fc_{output_id}"
      );"""
    ))
    return f"output_{output_id}"

  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    input_variable = run(self.input)
    print("zero points fc", self.weight_zero_point, self.output_zero_point)
    output = fc(input_variable, self.weight_data, self.bias_data, output_scale=self.multipliers, weight_zero_point=self.weight_zero_point, out_zero_point=self.output_zero_point)
    return output