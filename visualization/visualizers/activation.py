# -*- coding: utf-8 -*-


class ActivationVisualizer(object):

    def __init__(self, context):
        self.context = context

    def get_activation(self, tensor, index, input_image):
        activation = self.context.calculate(tensor, input_image=input_image)
        if activation.ndim == 2:
            return activation[:, index]
        elif activation.ndim == 3:
            return activation[:, :, index]
        elif activation.ndim == 4:
            return activation[0, :, :, index]
