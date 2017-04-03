# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from math import ceil


# TODO: This should br done individually for each feature:
# To examine a given convnet activation, we set all other activations in the layer to zero and pass
# the feature maps as input to the attached deconvnet layer. Then we successively (i) unpool,
# (ii) rectify and (iii) filter to reconstruct the activity in the layer beneath that gave rise to
# the chosen activation. This is then repeated until input pixel space is reached.
class ZeilerEtAlVisualizer(object):
    def __init__(self, context):
        self._context = context

    def reconstruct_input(self, tensor_name, channel_index, input_image, log=False):
        output_tensor = self._context.get_tensor(tensor_name)

        tensor = output_tensor
        current_output = self._context.calculate(tensor, input_image=input_image)
        # Zero out all the channel except the give one.
        to_be_zeroed = np.ones(current_output.shape).astype(np.bool)
        to_be_zeroed[[slice(None)] * (current_output.ndim - 1) + [slice(channel_index, channel_index + 1)]] = False
        current_output[to_be_zeroed] = 0

        while tensor is not None:
            input_tensors = list(tensor.op.inputs)
            current_inputs = self._context.calculate(input_tensors, input_image=input_image)

            if log:
                print('Processing {}'.format(tensor.op.type))

            # TODO: Add support for 'add', 'sub', 'mul', and 'split' operations.
            if tensor.op.type == 'ExpandDims':
                current_output = current_output.reshape(current_inputs[0].shape)

            elif tensor.op.type == 'Split':
                raise NotImplementedError()

            elif tensor.op.type == 'Concat':
                # The scalar input specifies the axis.
                axis = next(i for i in current_inputs if i.ndim == 0)
                begin, end = 0, 0
                for t, i in zip(input_tensors, current_inputs):
                    if i.ndim == 0:
                        continue
                    if t.op.type == 'Const':
                        begin += i.shape[axis]
                    else:
                        end = begin + i.shape[axis]
                        break
                index = [slice(None)] * current_output.ndim
                index[axis] = slice(begin, end)
                current_output = current_output[index]

            elif tensor.op.type == 'MaxPool':
                # TODO: Add support for 'VALID' padding.
                if tensor.op.get_attr('padding').decode('utf-8') != 'SAME':
                    raise RuntimeError('Operation with unsupported padding: {}'.format(tensor.op.name))
                # TODO: Add support for 'NCHW' data format.
                if tensor.op.get_attr('data_format').decode('utf-8') != 'NHWC':
                    raise RuntimeError('Operation with unsupported data format: {}'.format(tensor.op.name))

                ksize = tensor.op.get_attr('ksize')
                strides = tensor.op.get_attr('strides')
                # Pass first non-constant input.
                max_pool_input = current_inputs[next(i for i, t in enumerate(input_tensors) if t.op.type != 'Const')]
                current_output = self._max_unpool(current_output, max_pool_input, ksize, strides)

            elif tensor.op.type == 'Conv2D':
                # TODO: Add support for 'NCHW' data format.
                if tensor.op.get_attr('data_format').decode('utf-8') != 'NHWC':
                    raise RuntimeError('Operation with unsupported data format: {}'.format(tensor.op.name))

                has_fixed_shape = list(map(lambda t: t.shape.is_fully_defined(), input_tensors))
                if has_fixed_shape.count(True) != 1:
                    raise RuntimeError('Ambiguous kernel for:{}'.format(tensor.op.name))
                kernel = current_inputs[has_fixed_shape.index(True)]
                strides = tensor.op.get_attr('strides')
                padding = tensor.op.get_attr('padding').decode('utf-8')
                # Pass first non-constant input.
                conv2d_input = current_inputs[next(i for i, t in enumerate(input_tensors) if t.op.type != 'Const')]
                current_output = self._deconv2d(current_output, conv2d_input, kernel, strides, padding)

            elif tensor.op.type == 'BiasAdd':
                # TODO: Add support for 'NCHW' data format.
                if tensor.op.get_attr('data_format').decode('utf-8') != 'NHWC':
                    raise RuntimeError('Operation with unsupported data format: {}'.format(tensor.op.name))

                has_fixed_shape = list(map(lambda t: t.shape.is_fully_defined(), input_tensors))
                if has_fixed_shape.count(True) != 1:
                    raise RuntimeError('Ambiguous bias for: {}'.format(tensor.op.name))
                bias = current_inputs[has_fixed_shape.index(True)]
                current_output = current_output - bias

            elif tensor.op.type == 'Add':
                raise NotImplementedError()

            elif tensor.op.type == 'Sub':
                raise NotImplementedError()

            elif tensor.op.type == 'Mul':
                raise NotImplementedError()

            elif tensor.op.type == 'Relu':
                current_output = np.maximum(current_output, 0)

            # Ignore the following operations (operation input and output size should match).
            elif tensor.op.type in ['LRN']:
                pass

            elif tensor.op.type == 'Const':
                raise RuntimeError('Constant operation: {}'.format(tensor.op.name))

            else:
                raise RuntimeError('Irreversible operation: {}'.format(tensor.op.name))

            for i, input_tensor in enumerate(input_tensors):
                if input_tensor == self._context.input_tensor:
                    return current_output

            tensor = next(
                filter(lambda t: t.op.type != 'Const', input_tensors),
                None
            )

        raise RuntimeError('Unable to reach `input`')

    @staticmethod
    def _max_unpool(current_output, max_pool_input, ksize, strides):
        paddings = []
        for dsize, stride in zip(max_pool_input.shape, strides):
            total_padding = ceil(dsize / stride) * stride - dsize
            padding_after = int(total_padding / 2)
            padding_before = int(total_padding - padding_after)
            paddings.append((padding_before, padding_after))
        max_pool_input = np.pad(max_pool_input, paddings, mode='constant')

        recon_input = np.zeros(max_pool_input.shape).astype(current_output.dtype)
        begin = np.zeros(4, dtype=np.int)
        end = begin + ksize
        index = [0] * current_output.ndim
        while True:
            ranges = np.ix_(*[range(b, e) for b, e in zip(begin, end)])
            slice_max = np.max(max_pool_input[ranges])

            args = np.argwhere(max_pool_input[ranges] == slice_max) + begin
            indexes = tuple([np.array(args[:, i]) for i in range(args.shape[1])])

            recon_input[indexes] = current_output[tuple(index)]
            i = len(begin)
            while True:
                i -= 1
                begin[i] += strides[i]
                end[i] = min(begin[i] + ksize[i], max_pool_input.shape[i])
                index[i] += 1
                if begin[i] >= max_pool_input.shape[i]:
                    begin[i] = 0
                    end[i] = min(ksize[i], max_pool_input.shape[i])
                    index[i] = 0
                    continue
                break

            if i < 0:
                unpaddings_index = []
                for padding in paddings:
                    begin = padding[0]
                    if padding[1] == 0:
                        end = None
                    else:
                        end = -padding[1]
                    unpaddings_index.append(slice(begin, end))

                recon_input = recon_input[unpaddings_index]
                return recon_input

    @staticmethod
    def _deconv2d(current_output, conv2d_input, kernel, strides, padding):
        with tf.Graph().as_default():
            session = tf.Session()
            reconstructed_conv2d_input = session.run(
                tf.nn.conv2d_transpose(
                    tf.constant(current_output),
                    tf.constant(kernel),
                    conv2d_input.shape,
                    strides=strides,
                    padding=padding
                )
            )
            session.close()
        return reconstructed_conv2d_input
