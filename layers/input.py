from dataclasses import dataclass
from io import UnsupportedOperation
import textwrap
from typing import Callable, List

from layers.layer import Layer
from layers.id_gen import IdGenerator
import numpy as np
@dataclass
class InputImage(Layer):
  input_width: int
  input_height: int
  input_depth: int

  output_scale: float

  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    output_id = id_gen.get_id(self)
    data_buffer.append(textwrap.dedent(f"""\
      #define output_{output_id}_dim {self.input_height}*{self.input_width}*{self.input_depth}
      elem_t output_{output_id}[{self.input_height * self.input_width}][{self.input_depth}];
    """))
    return f"output_{output_id}"

  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    raise UnsupportedOperation("Need to pass in inital input image data through the cache")

@dataclass
class Input1D(Layer):
  input_size: int

  output_scale: float

  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    output_id = id_gen.get_id(self)
    data_buffer.append(textwrap.dedent(f"""\
      #define output_{output_id}_dim {self.input_size}
      elem_t output_{output_id}[{self.input_size}][1];
    """))
    return f"output_{output_id}"

  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    raise UnsupportedOperation("Need to pass in inital input image data through the cache")
