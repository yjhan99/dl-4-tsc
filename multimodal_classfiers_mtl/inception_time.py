# based on https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py


import keras

# from multimodal_classfiers.classifier import Classifier
from multimodal_classfiers_finetuning.classifier_finetuning import Classifier


class ClassifierInception(Classifier):

    def __init__(self, output_directory, input_shapes, nb_classes, verbose=False, build=True,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, hyperparameters=None,
                 model_init=None):
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.bottleneck_size = 32

        super().__init__(output_directory, input_shapes, nb_classes, verbose=verbose, hyperparameters=hyperparameters,
                         model_init=model_init)

    def _inception_module(self, input_tensor, filters, kernel_size, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=filters,
                                                 kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPooling1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layers = []
        channel_outputs = []

        kernel_sizes = hyperparameters.kernel_sizes if hyperparameters and hyperparameters.kernel_sizes else 100 * [
            self.kernel_size]
        filters = hyperparameters.kernel_sizes if hyperparameters and hyperparameters.kernel_sizes else 100 * [
            self.nb_filters]
        for channel_id, input_shape in enumerate(input_shapes):

            input_layer = keras.layers.Input(input_shape)
            input_layers.append(input_layer)

            x = input_layer
            input_res = input_layer

            for d in range(self.depth):
                x = self._inception_module(x, filters[channel_id], kernel_sizes[channel_id])

                if self.use_residual and d % 3 == 2:
                    x = self._shortcut_layer(input_res, x)
                    input_res = x

            gap_layer = keras.layers.GlobalAveragePooling1D()(x)
            channel_outputs.append(gap_layer)

        flatten_layer = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else \
            channel_outputs[0]
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model