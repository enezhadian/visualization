# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from .optimizer import OptimizationBasedVisualizer


class MahendranEtAlVisualizer(OptimizationBasedVisualizer):

    def __init__(self, context, l2_decay_weight=2e-1, total_variation_weight=5e-1):
        self._context = context
        self._regularizer = lambda x: (self._l2_decay(l2_decay_weight)(x))
        self._total_variation_weight = total_variation_weight

    def reconstruct_input(self, tensor_name, channel_index, input_image, iterations=200, step=1.0, log_per=None):
        score_tensor = self._score_tensor_builder(tensor_name, channel_index, input_image)
        # TODO: Get the size from context.
        noise_input = np.random.uniform(size=(224, 224, 3))

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

    def _score_tensor_builder(self, tensor_name, channel_index, input_image):
        reference_activation = self._context.calculate(
            tensor_name,
            input_image=input_image
        )[0, :, :, channel_index]
        activation_tensor = self._context.get_tensor(tensor_name)[0, :, :, channel_index]

        with self._context.graph.as_default():
            loss_tensor = tf.nn.l2_loss(activation_tensor - reference_activation)
            regularization_tensor = tf.image.total_variation(tf.expand_dims(activation_tensor, -1))

            # Devide by a big value to prevent overflow.
            score_tensor = -(loss_tensor + self._total_variation_weight * regularization_tensor) / 1e4
        return score_tensor

    @staticmethod
    def _l2_decay(weight):
        def regularizer(array):
            return (1 - weight) * array
        return regularizer
