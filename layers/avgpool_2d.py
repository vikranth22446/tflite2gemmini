from dataclasses import dataclass
import textwrap
from typing import Callable, List

from layers.layer import Layer
from layers.id_gen import IdGenerator
from simulation import avg_pool2d
import numpy as np

@dataclass
class AvgPool2D(Layer):
  input: Layer

  input_width: int
  input_height: int
  input_depth: int

  output_width: int
  output_height: int
  output_depth: int

  @property
  def output_scale(self):
    return self.input.output_scale

  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: IdGenerator, generate: Callable[["Layer"], str]) -> str:
    input_variable = generate(self.input)

    # we don't support proper pooling in the runtime impl
    assert self.output_width == 1
    assert self.output_height == 1

    output_id = id_gen.get_id(self)

    data_buffer.append(textwrap.dedent(f"""\
      elem_t output_{output_id}[{self.output_height} * {self.output_width}][{self.output_depth}];
    """))

    runtime_buffer.append(textwrap.dedent(f"""\
      compute_average_pooling(
        {self.input_width}, {self.input_depth},
        {input_variable},
        output_{output_id}
      );"""
    ))
    return f"output_{output_id}"

  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    input_variable = run(self.input)

    # TODO set pooling shape
    output = avg_pool2d(input_variable)
    return output