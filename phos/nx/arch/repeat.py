from ..module import Module
from .sequence import Sequence


class Repeat(Sequence):
    def __init__(self, count, new_layer):
        layers = [new_layer() for i in range(count)]
        super().__init__(*layers)
