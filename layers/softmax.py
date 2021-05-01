
from dataclasses import dataclass
import textwrap
from typing import Callable, List

from layers.layer import Layer
from layers.id_gen import IdGenerator
import numpy as np
from simulation import softmax_layer
@dataclass
class Softmax(Layer):
  input: Layer
  output_size: int
  out_scale: float
  output_zero_point:int

  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    input_variable = generate(self.input)

    output_id = id_gen.get_id(self)

    data_buffer.append(textwrap.dedent(f"""\
      elem_t output_{output_id}[{self.output_size}];"""
    ))

    runtime_buffer.append(textwrap.dedent(f"""\
      softmax({self.output_size}, {input_variable}, output_{output_id});"""
    ))

    return f"output_{output_id}"
  
  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    input_variable = run(self.input)

    output = softmax_layer(input_variable, self.input.output_zero_point, self.input.output_scale, self.output_zero_point, self.out_scale)
    return output