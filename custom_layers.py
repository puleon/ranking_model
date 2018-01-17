from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
from keras import initializers


class Bias(Layer):
    def __init__(self, bias_initializer='zeros', **kwargs):
        super(Bias, self).__init__(**kwargs)
        self.bias_initializer = initializers.get(bias_initializer)
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.bias = self.add_weight(shape=(1,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    trainable=True)
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.bias_add(inputs, self.bias)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = 1
        return tuple(output_shape)
