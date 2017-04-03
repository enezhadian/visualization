# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
from abc import ABCMeta, abstractmethod


class OptimizationBasedVisualizer(metaclass=ABCMeta):

    @abstractmethod
    def reconstruct_input(self, *args, **kwargs):
        pass

    @staticmethod
    def _gradient_ascent(session, score_t, input_t, initial_image, regularizer=None,
                         iterations=20, step=1.0, log_per=None):
        gradient_t = tf.gradients(score_t, input_t)[0]

        max_image = None
        max_score = -1000000

        image = initial_image.copy()
        for i in range(iterations):
            gradient, score = session.run([gradient_t, score_t], {input_t: image})
            if score >= max_score:
                max_score = score
                max_image = image
            image = image + gradient * step
            if regularizer is not None:
                image = regularizer(image)
            if log_per and i % log_per == 0:
                print('Iteration: {} => Score: {}'.format(i, score))
                sys.stdout.flush()
        return max_image
