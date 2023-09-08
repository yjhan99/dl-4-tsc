import keras

from multimodal_classfiers.classifier import Classifier


class ClassifierTimeCnn(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layers = []

        filters_multipliers = [1] * len(input_shapes)
        kernel_size_multipliers = [1] * len(input_shapes)

        if hyperparameters:
            filters_multipliers = hyperparameters.filters_multipliers
            kernel_size_multipliers = hyperparameters.kernel_size_multipliers

        channel_outputs = []
        for channel_id, input_shape in enumerate(input_shapes):
            padding = 'valid'

            input_layer = keras.layers.Input(input_shape)
            input_layers.append(input_layer)

            kernel_size = int(kernel_size_multipliers[channel_id] * 7)

            conv1 = keras.layers.Conv1D(filters=int(filters_multipliers[channel_id] * 6), kernel_size=kernel_size,
                                        padding=padding, activation='relu')(input_layer)
            conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

            conv2 = keras.layers.Conv1D(filters=int(filters_multipliers[channel_id] * 12), kernel_size=kernel_size,
                                        padding=padding, activation='relu')(conv1)
            conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

            flatten_layer = keras.layers.Flatten()(conv2)
            channel_outputs.append(flatten_layer)

        flatten_layer = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else \
            channel_outputs[0]
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model
