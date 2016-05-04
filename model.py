from lasagne.layers import InputLayer, InverseLayer

from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer


def build_model():
    net = {}
    net['input'] = InputLayer((None, params.MAX_LENGTH))
    net['conv1'] = ConvLayer(net['input'], num_filters=32, filter_size=100)
    net['conv2'] = ConvLayer(net['conv1'], num_filters=64, filter_size=100)
    net['pool1'] = PoolLayer(net['conv2'], pool_size=2)
    net['conv3'] = ConvLayer(net['pool1'], num_filters=128, filter_size=50)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=256, filter_size=50)
    net['pool2'] = PoolLayer(net['conv4'], pool_size=2)

    # Deconv
    net['pool2/inverse'] = InverseLayer(net['pool2'], net['conv4'])
    net['conv4/inverse'] = InverseLayer(net['conv4'], net['conv3'])
    net['conv3/inverse'] = InverseLayer(net['conv3'], net['pool1'])
    net['pool1/inverse'] = InverseLayer(net['pool1'], net['conv2'])
    net['conv2/inverse'] = InverseLayer(net['conv2'], net['conv1'])
    net['conv1/inverse'] = InverseLayer(net['conv1'], net['input'])

    net['prob'] = net['conv1/inverse']

    # Network should end with `net['prob']` layer

    return net
