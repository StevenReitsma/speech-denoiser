from lasagne import nonlinearities
from lasagne.layers import InputLayer, InverseLayer, DenseLayer, ReshapeLayer, get_output_shape, get_all_layers, dropout, \
    SliceLayer

from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer
from lasagne.layers import TransposedConv2DLayer

from params import params


def inverse_dense_layer(input_layer, original_layer, orig_shape):
    return ReshapeLayer(dropout(DenseLayer(input_layer, num_units=orig_shape[1] * orig_shape[2],
                                           nonlinearity=nonlinearities.leaky_rectify, W=original_layer.input_layer.W.T), 0.5),
                        (-1, orig_shape[1], orig_shape[2]))


def inverse_convolution_strided_layer(input_layer, original_layer):
    return ReshapeLayer(SliceLayer(
        TransposedConv2DLayer(ReshapeLayer(input_layer, (-1, original_layer.output_shape[1], 1, original_layer.output_shape[2])),
                              original_layer.input_layer.num_filters, (1, original_layer.filter_size[0]),
                              stride=(1, original_layer.stride[0]), crop=(0, 0), flip_filters=original_layer.flip_filters, nonlinearity=nonlinearities.leaky_rectify),
        indices=slice(None, -1), axis=-1),
                        (-1, original_layer.input_shape[1], original_layer.input_shape[2]))


def inverse_convolution_layer(input_layer, original_layer):
    return ConvLayer(input_layer, num_filters=original_layer.input_layer.num_filters, filter_size=original_layer.filter_size,
                     nonlinearity=nonlinearities.leaky_rectify, pad='same')

def inverse_convolution_layer_2(input_layer, original_layer):
    return ConvLayer(input_layer, num_filters=original_layer.input_layer.num_filters, filter_size=1,
                     nonlinearity=nonlinearities.leaky_rectify, pad='same')


def build_model_dense(input_shape, input_var):
    net = {}
    net['input'] = InputLayer(input_shape, input_var=input_var)
    net['input'].num_filters = input_shape[1]
    net['conv1'] = ConvLayer(net['input'], num_filters=256, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv2'] = ConvLayer(net['conv1'], num_filters=256, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv2/reshape'] = ReshapeLayer(net['conv2'], (-1, net['conv2'].output_shape[1] * net['conv2'].output_shape[2]))
    net['dense'] = dropout(DenseLayer(net['conv2/reshape'], num_units=1024, nonlinearity=nonlinearities.leaky_rectify), 0.5)

    net['dense/inverse'] = inverse_dense_layer(net['dense'], net['dense'], net['conv2'].output_shape)
    net['conv2/inverse'] = inverse_convolution_layer(net['dense/inverse'], net['conv2'])
    net['conv1/inverse'] = inverse_convolution_layer(net['conv2/inverse'], net['conv1'])
    net['conv0/inverse'] = ConvLayer(net['conv1/inverse'], num_filters=input_shape[1], filter_size=1,nonlinearity=nonlinearities.linear, pad='same')
    net['prob'] = net['conv0/inverse']
    for layer in get_all_layers(net['prob']):
        print layer
        print layer.output_shape
    return net

def build_model_small(input_shape, input_var):
    net = {}
    net['input'] = InputLayer(input_shape, input_var=input_var)
    net['input'].num_filters = input_shape[1]
    net['conv1'] = ConvLayer(net['input'], num_filters=256, filter_size=11, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv2'] = ConvLayer(net['conv1'], num_filters=256, filter_size=7, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv3'] = ConvLayer(net['conv2'], num_filters=396, filter_size=5, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv5'] = ConvLayer(net['conv4'], num_filters=1024, filter_size=1, nonlinearity=nonlinearities.leaky_rectify,pad='same')
    net['conv5/inverse'] = inverse_convolution_layer(net['conv5'], net['conv5'])
    net['conv4/inverse'] = inverse_convolution_layer(net['conv5/inverse'], net['conv4'])
    net['conv3/inverse'] = inverse_convolution_layer(net['conv4/inverse'], net['conv3'])
    net['conv2/inverse'] = inverse_convolution_layer(net['conv3/inverse'], net['conv2'])
    net['conv1/inverse'] = inverse_convolution_layer(net['conv2/inverse'], net['conv1'])
    net['conv0/inverse'] = ConvLayer(net['conv1/inverse'], num_filters=input_shape[1], filter_size=1,nonlinearity=nonlinearities.linear, pad='same')
    net['prob'] = net['conv0/inverse']
    for layer in get_all_layers(net['prob']):
        print layer
        print layer.output_shape
    return net


def build_model(input_shape, input_var, dense=True):
    net = {}
    net['input'] = InputLayer(input_shape, input_var=input_var)
    net['input'].num_filters = input_shape[1]
    net['conv1'] = ConvLayer(net['input'], num_filters=128, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv2'] = ConvLayer(net['conv1'], num_filters=256, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['pool1'] = ConvLayer(net['conv2'], num_filters=256, filter_size=3, stride=2, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv3'] = ConvLayer(net['pool1'], num_filters=512, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['pool2'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, stride=2, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    if dense:
        net['dense'] = dropout(DenseLayer(net['pool2'], num_units=1024, nonlinearity=nonlinearities.leaky_rectify), 0.5)
        # Deconv
        net['dense/inverse'] = inverse_dense_layer(net['dense'], net['dense'], net['pool2'].output_shape)
        net['pool2/inverse'] = inverse_convolution_strided_layer(net['dense/inverse'], net['pool2'])
    else:
        net['pool2/inverse'] = inverse_convolution_strided_layer(net['pool2'], net['pool2'])
    net['conv3/inverse'] = inverse_convolution_layer(net['pool2/inverse'], net['conv3'])
    net['pool1/inverse'] = inverse_convolution_strided_layer(net['conv3/inverse'], net['pool1'])
    net['conv2/inverse'] = inverse_convolution_layer(net['pool1/inverse'], net['conv2'])
    net['conv1/inverse'] = inverse_convolution_layer(net['conv2/inverse'], net['conv1'])
    net['conv0/inverse'] = ConvLayer(net['conv1/inverse'], num_filters=input_shape[1], filter_size=1, nonlinearity=nonlinearities.linear, pad='same')

    net['prob'] = net['conv0/inverse']

    for layer in get_all_layers(net['prob']):
        print layer
        print layer.output_shape
    return net
