from lasagne import nonlinearities
from lasagne.layers import InputLayer, InverseLayer

from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer

from params import params


def build_model(input_shape, input_var):
    net = {}
    net['input'] = InputLayer(input_shape, input_var=input_var)
    net['conv1'] = ConvLayer(net['input'], num_filters=32, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv2'] = ConvLayer(net['conv1'], num_filters=64, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['pool1'] = PoolLayer(net['conv2'], pool_size=2)
    net['conv3'] = ConvLayer(net['pool1'], num_filters=128, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv4'] = ConvLayer(net['conv3'], num_filters=256, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['pool2'] = PoolLayer(net['conv4'], pool_size=2)

    # Deconv
    net['pool2/inverse'] = InverseLayer(net['pool2'], net['pool2'])
    net['conv4/inverse'] = InverseLayer(net['pool2/inverse'], net['conv4'])
    net['conv3/inverse'] = InverseLayer(net['conv4/inverse'], net['conv3'])
    net['pool1/inverse'] = InverseLayer(net['conv3/inverse'], net['pool1'])
    net['conv2/inverse'] = InverseLayer(net['pool1/inverse'], net['conv2'])
    net['conv1/inverse'] = InverseLayer(net['conv2/inverse'], net['conv1'])
    #net['conv1/inverse'] = InverseLayer(net['conv1/inverse'], net['input'])

    net['prob'] = net['conv1/inverse']

    # Network should end with `net['prob']` layer

    return net
