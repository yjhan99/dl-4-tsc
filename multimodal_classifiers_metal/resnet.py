import keras

class Resnet(keras.Model):
    def __init__(self, input_shapes, nb_classes, hyperparameters=None):
        super(Resnet, self).__init__()
        self.input_shapes = input_shapes
        self.nb_classes = nb_classes
        self.hyperparameters = hyperparameters

        kernel_size_multipliers = [1] * len(input_shapes)
        filters = [64] * len(input_shapes)
        depth = 3

        self.input_layers = []
        channel_outputs = []

        for channel_id, input_shape in enumerate(input_shapes):
            current_layer = keras.layers.Input(shape=input_shape, name=f"input_{channel_id}")
            self.input_layers.append(current_layer)

            for i_depth in range(depth - 1):
                mult = 2 if i_depth > 0 else 1
                current_layer = self.build_bloc(int(mult * filters[channel_id]), kernel_size_multipliers[channel_id], current_layer)

            # Last block
            conv_x = keras.layers.Conv1D(filters=int(filters[channel_id] * 2), kernel_size=int(kernel_size_multipliers[channel_id] * 8), padding='same')(current_layer)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)

            conv_y = keras.layers.Conv1D(filters=int(filters[channel_id] * 2), kernel_size=int(kernel_size_multipliers[channel_id] * 5), padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)

            conv_z = keras.layers.Conv1D(filters=int(filters[channel_id] * 2), kernel_size=int(kernel_size_multipliers[channel_id] * 3), padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)

            shortcut_y = keras.layers.Conv1D(filters=filters[channel_id] * 2, kernel_size=1, padding='same')(current_layer) if depth == 2 else current_layer
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

            output_block_3 = keras.layers.add([shortcut_y, conv_z])
            output_block_3 = keras.layers.Activation('relu')(output_block_3)

            # Global Average Pooling
            gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
            channel_outputs.append(gap_layer)

        # Concatenate or single channel output
        flatten_layer = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]
        
        # Output layer
        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(flatten_layer)

        self.model = keras.models.Model(inputs=self.input_layers, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def build_bloc(self, n_feature_maps, kernel_size_multiplier, input_layer):
        # Convolutional layers
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=int(kernel_size_multiplier * 8), padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=int(kernel_size_multiplier * 5), padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=int(kernel_size_multiplier * 3), padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # Shortcut connection
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        return keras.layers.Activation('relu')(output_block_1)

    def call(self, inputs):
        return self.model(inputs)