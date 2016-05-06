from lasagne import nonlinearities
from lasagne.layers import InputLayer, InverseLayer, DenseLayer, ReshapeLayer, get_output_shape, get_all_layers

from lasagne.layers import Pool1DLayer as PoolLayer
from lasagne.layers import Conv1DLayer as ConvLayer

from params import params


def build_model(input_shape, input_var):
    net = {}
    net['input'] = InputLayer(input_shape, input_var=input_var)
    net['conv1'] = ConvLayer(net['input'], num_filters=128, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    net['conv2'] = ConvLayer(net['conv1'], num_filters=256, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    #net['pool1'] = PoolLayer(net['conv2'], pool_size=2)
    #net['pool1'] = ConvLayer(net['conv2'], num_filters=64, filter_size=3, stride=2, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    #net['conv3'] = ConvLayer(net['pool1'], num_filters=128, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    #net['conv4'] = ConvLayer(net['pool1'], num_filters=256, filter_size=3, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    #net['pool2'] = PoolLayer(net['conv4'], pool_size=2)
    #net['pool2'] = ConvLayer(net['conv4'], num_filters=256, filter_size=3, stride=2, nonlinearity=nonlinearities.leaky_rectify, pad='same')
    #net['dense'] = DenseLayer(net['conv1'], num_units=256, nonlinearity=nonlinearities.leaky_rectify)

    # Deconv
    orig_shape = get_output_shape(net['conv1'])
    #net['dense/inverse'] = ReshapeLayer(DenseLayer(net['dense'], num_units=orig_shape[1]*orig_shape[2], nonlinearity=nonlinearities.leaky_rectify, W=net['dense'].W.T),
                                        #(-1, orig_shape[1], orig_shape[2]))

    net['conv2/inverse'] = ConvLayer(net['conv2'], num_filters=net['conv1'].num_filters,
                                     filter_size=net['conv2'].filter_size,
                                     nonlinearity=nonlinearities.leaky_rectify, pad='same')

    net['conv1/inverse'] = ConvLayer(net['conv2/inverse'], num_filters=input_shape[1], filter_size=net['conv1'].filter_size,
                                     nonlinearity=nonlinearities.linear, pad='same')

    net['conv0/inverse'] = ConvLayer(net['conv1/inverse'], num_filters=input_shape[1],
                                     filter_size=1, nonlinearity=nonlinearities.linear, pad='same')


    #net['dense/inverse'] = InverseLayer(net['dense'], net['dense'])
    #net['pool2/inverse'] = InverseLayer(net['dense/inverse'], net['pool2'])
    #net['conv4/inverse'] = InverseLayer(net['pool2/inverse'], net['conv4'])
    #net['conv3/inverse'] = InverseLayer(net['conv4/inverse'], net['conv3'])
    #net['pool1/inverse'] = InverseLayer(net['conv3/inverse'], net['pool1'])
    #net['conv2/inverse'] = InverseLayer(net['pool1/inverse'], net['conv2'])
    #net['conv1/inverse'] = InverseLayer(net['conv2/inverse'], net['conv1'])
    #net['conv1/inverse'] = InverseLayer(net['conv1/inverse'], net['input'])

    net['prob'] = net['conv0/inverse']

    for layer in get_all_layers(net['prob']):
        print layer
        print get_output_shape(layer)

    # Network should end with `net['prob']` layer

    return net
