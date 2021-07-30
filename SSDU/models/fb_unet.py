import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

def deconv2d(input_tensor, filter_size, output_h, output_w, out_channels, in_channels, name, strides = [1, 1, 1, 1]):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_h, output_w, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    W = tf.compat.v1.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, W, out_shape, strides, padding='SAME')
    return h1

def conv2d(input_tensor, conv_filter, name, strides=(1, 1), padding="SAME"):
    W = tf.compat.v1.get_variable(name, shape=conv_filter)
    #W = tf.compat.v1.get_variable(name, shape=conv_filter, initializer=tf.random_normal_initializer(0, 0.05))
    x = tf.nn.conv2d(input_tensor, W, strides=strides, padding=padding, name=name)
    return tf.nn.relu(x)

def max_pool(x):
    return tf.nn.max_pool(x, ksize=2, strides=1, padding='SAME')

def U_net(X, out_shape):

    #[filter_height, filter_width, in_channels, out_channels]
    conv_filters = dict([('Y0', (1, 1, 2, 32)), 
    ('Y2', (3, 3, 32, 64)), 
    ('Y3', (3, 3, 64, 128))])
    # ('Y2_deconv', (1, 1, 128, 128)), 
    # ('Y1_deconv', (2, 2, 128, 64)),
    # ('Y0_deconv', (2, 2, 64, 32)), 
    # ('logits_deconv', (1, 1, 32, 2)),])

    with tf.compat.v1.variable_scope('conv1'):
        net = conv2d(X, conv_filters["Y0"], "Y0") #128
        net = max_pool(net)

    with tf.compat.v1.variable_scope('conv2'):
        net = conv2d(net, conv_filters["Y2"], "Y2", strides=(2, 2)) #64
        net = max_pool(net)

    with tf.compat.v1.variable_scope('conv3'):
        net = conv2d(net, conv_filters["Y3"], "Y3", strides=(2, 2)) #32
        net = max_pool(net)

    with tf.compat.v1.variable_scope('deconv1'):
        net = deconv2d(net, 1, 160, 93, 128, 128, "Y2_deconv") # 32
        net = tf.nn.relu(net)
    
    with tf.compat.v1.variable_scope('deconv2'):
        net = deconv2d(net, 2, 320, 186, 64, 128, "Y1_deconv", strides=[1, 2, 2, 1]) # 64
        net = tf.nn.relu(net)
    
    with tf.compat.v1.variable_scope('deconv3'):
        net = deconv2d(net, 2, 640, out_shape, 32, 64, "Y0_deconv", strides=[1, 2, 2, 1]) # 128
        net = tf.nn.relu(net)
    
    with tf.compat.v1.variable_scope('last'):
        logits = deconv2d(net, 1, 640, out_shape, 2, 32, "logits_deconv") # 128
    return logits

