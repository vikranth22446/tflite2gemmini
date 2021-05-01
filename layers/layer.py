from abc import ABC, abstractmethod
from typing import Callable, Dict, List
import numpy as np
import layers

class Layer(ABC):
  @abstractmethod
  def generate_gemmini(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: "layers.id_gen.IdGenerator", generate: Callable[["Layer"], str]) -> str:
    pass

  def generate_gemmini_cached(self, data_buffer: List[str], runtime_buffer: List[str], id_gen: "layers.id_gen.IdGenerator", cache: Dict[int, str]) -> str:
    if id(self) not in cache:
      cache[id(self)] = self.generate_gemmini(data_buffer, runtime_buffer, id_gen, lambda layer: layer.generate_gemmini_cached(data_buffer, runtime_buffer, id_gen, cache))
    return cache[id(self)]

  @abstractmethod
  def simulate_python(self, run: Callable[["Layer"], np.ndarray]) -> np.ndarray:
    pass

  def simulate_python_cached(self, execute_order: List["Layer"], cache: Dict[int, np.ndarray]) -> np.ndarray:
    if id(self) not in cache:
      cache[id(self)] = self.simulate_python(lambda layer: layer.simulate_python_cached(execute_order, cache))
      execute_order.append(self)
    return cache[id(self)]
