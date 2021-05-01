from typing import Dict
from layers.layer import Layer

class IdGenerator(object):
  next_id = 0
  existing_ids: Dict[int, int] = {}

  def get_id(self, layer: Layer) -> int:
    if id(layer) not in self.existing_ids:
      self.existing_ids[id(layer)] = self.next_id
      self.next_id += 1
    return self.existing_ids[id(layer)]
