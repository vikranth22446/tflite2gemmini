from dataclasses import dataclass
import textwrap
from typing import Callable, List

import numpy as np

from layers.layer import Layer
from layers.id_gen import IdGenerator

@dataclass
class Reshape(Layer):
  input: Layer
  output_size: int

  @property
  def output_scale(self):
    return self.input.output_scale

  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    input_variable = generate(self.input)

    output_id = id_gen.get_id(self)

    data_buffer.append(textwrap.dedent(f"""\
      elem_t (*output_{output_id})[{self.output_size}] = {input_variable};"""
    ))

    runtime_buffer.append(textwrap.dedent(f"""\
      elem_t (*output_{output_id})[{self.output_size}] = {input_variable};"""
    ))
    return f"output_{output_id}"
  
  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    input_variable = run(self.input)
    output = input_variable.reshape(self.output_size)
    return output