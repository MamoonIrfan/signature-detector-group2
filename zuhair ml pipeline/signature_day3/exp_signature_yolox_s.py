"""Export architecture exactly matches the single-class training network."""
from yolox.exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 1
        self.input_size = (640, 640)
        self.test_size = (640, 640)
