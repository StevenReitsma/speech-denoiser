from lasagne import nonlinearities
from lasagne.layers import InputLayer, InverseLayer, DenseLayer, ReshapeLayer, get_output_shape, get_all_layers, dropout, \
    SliceLayer, batch_norm

from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer
from lasagne.layers import TransposedConv2DLayer

from params import params


def build_model_small(input_shape, input_var):
    net = {}
    net['input'] = InputLayer(input_shape, input_var=input_var)
    net['input'].num_filters = input_shape[1]
    net['conv1'] = batch_norm(ConvLayer(net['input'], num_filters=256, filter_size=11, nonlinearity=nonlinearities.leaky_rectify, pad='same'))
    net['pool1'] = dropout(PoolLayer(net['conv1'], 2, mode='max'), 0.5)
    net['conv2'] = batch_norm(ConvLayer(net['pool1'], num_filters=256, filter_size=7, nonlinearity=nonlinearities.leaky_rectify, pad='same'))
    net['pool2'] = dropout(PoolLayer(net['conv2'], 2, mode='max'), 0.5)
    net['conv3'] = batch_norm(ConvLayer(net['pool2'], num_filters=396, filter_size=5, nonlinearity=nonlinearities.leaky_rectify, pad='same'))
    net['pool3'] = dropout(PoolLayer(net['conv3'], 2, mode='max'), 0.5)
    net['conv4'] = dropout(batch_norm(ConvLayer(net['pool3'], num_filters=512, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')), 0.5)
    net['conv5'] = dropout(batch_norm(ConvLayer(net['conv4'], num_filters=1024, filter_size=1, nonlinearity=nonlinearities.leaky_rectify,pad='same')), 0.5)
    net['dense1'] = dropout(batch_norm(DenseLayer(net['conv5'], num_units=1024, nonlinearity=nonlinearities.leaky_rectify)), 0.5)
    net['dense2'] = DenseLayer(net['dense1'], num_units=11, nonlinearity=nonlinearities.softmax)
    net['prob'] = net['dense2']
    for layer in get_all_layers(net['prob']):
        print layer
        print layer.output_shape
    return net

