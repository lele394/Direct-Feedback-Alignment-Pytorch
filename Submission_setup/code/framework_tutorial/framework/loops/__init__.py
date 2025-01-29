from .basic_dfa import train_basic
from .average_dfa import train_averaged
from .class_average import MNIST_train_class_averaged
from .class_average_with_lr_scheduler import MNIST_train_class_averaged_LR_scheduler

# Backprop train loop
from .backprop_train_loop import train as Backprop_train

# Special training loops
from .mixed_comparison_train_loop import train_mix
