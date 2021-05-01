from dataclasses import dataclass
import textwrap
from typing import Callable, List

import numpy as np

from utils import np2carray
from layers.layer import Layer
from layers.id_gen import IdGenerator
from simulation import residual_add

@dataclass
class Add(Layer):
  inputs: List[Layer]

  output_scale:float

  pot_scale_int16:bool # some attribute to support 16 bit activations
  followed_by_relu:bool

  output_width:int
  output_height:int
  output_depth:int

  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    input_variables = []
    for input in self.inputs:
        input_variables.append(generate(input))

    output_id = id_gen.get_id(self)

    data_buffer.append(textwrap.dedent(f"""\
      elem_t output_{output_id}[{self.output_height} * {self.output_width}][{self.output_depth}];"""
    ))

    runtime_buffer.append(textwrap.dedent(f"""\
      add({self.output_width}, {self.output_height}, {self.output_depth}, {self.output_scale}, {input_variables[0]},{input_variables[1]}, output_{output_id});"""
    ))

    return f"output_{output_id}"
  
  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    input_variables = []
    for input in self.inputs:
        input_variables.append(run(input))
    
    output = residual_add(input_variables[0], input_variables[1])
    return output