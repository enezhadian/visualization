# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from .optimizer import OptimizationBasedVisualizer


class ErhanEtAlVisualizer(OptimizationBasedVisualizer):

    def __init__(self, context, data_mean):
        self._context = context
        self._data_mean = data_mean

    def reconstruct_input(self, tensor_name, unit_index, iterations=200, step=1.0, log_per=None):
        score_tensor = self._score_tensor_builder(tensor_name, unit_index)
        # TODO: Get the size from context.
        noise_input = self._normalize(magnitude=self._data_mean)(np.random.uniform(size=(224, 224, 3)))

        reconstructed_input = self._gradient_ascent(
            self._context.session,
            score_tensor,
            self._context.input_tensor,
            noise_input,
            regularizer=self._normalize(magnitude=self._data_mean),
            iterations=iterations,
            step=step,
            log_per=log_per
        )
        return reconstructed_input

    def _score_tensor_builder(self, tensor_name, unit_index):
        score_tensor = self._context.get_tensor(tensor_name)[unit_index]
        return score_tensor

    @staticmethod
    def _normalize(magnitude):
        def regularizer(array):
            return (array / norm(array)) * magnitude
        return regularizer
