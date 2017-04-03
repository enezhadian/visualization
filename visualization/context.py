# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class Context(object):

    def __init__(self, pb_file_path, input_tensor_name='input'):
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        # Load graph definition.
        with self.graph.as_default():
            pb_file = tf.gfile.FastGFile(pb_file_path, 'rb')

            graph_def = tf.GraphDef()
            graph_def.ParseFromString(pb_file.read())

            self.input_tensor = tf.placeholder(np.float32, name=input_tensor_name)
            expanded_input_tensor = tf.expand_dims(self.input_tensor, 0)

            tf.import_graph_def(
                graph_def,
                {input_tensor_name: expanded_input_tensor}
            )

            pb_file.close()

    def get_output_tensor_names(self):
        tensor_names = []
        for op in self.graph.get_operations():
            if op.name.startswith('import/'):
                tensor_names.append(op.name[7:] + ':0')

        return tensor_names

    def get_tensor(self, tensor_name):
        return self.graph.get_tensor_by_name("import/{}".format(tensor_name))

    def calculate(self, tensors, input_image):
        if isinstance(tensors, str):
            tensors = 'import/' + tensors
        elif isinstance(tensors, list):
            for i, t in enumerate(tensors):
                if isinstance(t, str):
                    tensors[i] = 'import/' + t
        return self.session.run(tensors, feed_dict={self.input_tensor: input_image})

    def __del__(self):
        self.session.close()
