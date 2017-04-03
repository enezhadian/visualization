# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import norm
from scipy.ndimage.filters import gaussian_filter
from .optimizer import OptimizationBasedVisualizer


class YosinskiEtAlVisualizer(OptimizationBasedVisualizer):

    def __init__(self, context, data_mean, gaussian_blur_sigma=5e-1, clip_threshold=5e-2):
        self._context = context
        self._data_mean = data_mean
        self._regularizer = lambda x: self._gaussian_blur(gaussian_blur_sigma)(self._clip_small_norm(clip_threshold)(x))

    def reconstruct_input(self, tensor_name, unit_index, iterations=200, step=1.0, log_per=None):
        score_tensor = self._score_tensor_builder(tensor_name, unit_index)
        # TODO: Get the size from context.
        noise_input = np.random.uniform(size=(224, 224, 3)) + self._data_mean

        reconstructed_input = self._gradient_ascent(
            self._context.session,
            score_tensor,
            self._context.input_tensor,
            noise_input,
            regularizer=self._regularizer,
            iterations=iterations,
            step=step,
            log_per=log_per
        )
        return reconstructed_input

    def _score_tensor_builder(self, tensor_name, unit_index):
        score_tensor = self._context.get_tensor(tensor_name)[unit_index]
        return score_tensor

    @staticmethod
    def _gaussian_blur(sigma):
        def regularizer(array):
            return gaussian_filter(array, sigma=sigma)
        return regularizer

    @staticmethod
    def _clip_small_norm(threshold):
        def regularizer(array):
            array_norm = norm(array, axis=-1)
            threshold_norm = threshold * np.average(array_norm)
            array = array.copy()
            array[array_norm < threshold_norm] = 0.0
            return array
        return regularizer
