from tensorflow import keras


class FCN(keras.Model):
    def __init__(self, input_shapes, num_classes):
        super(FCN, self).__init__()
        self.input_shapes = input_shapes
        self.num_classes = num_classes
        self.filters_multipliers = [1] * len(input_shapes)
        self.kernel_size_multipliers = [1] * len(input_shapes)

        self.input_layers = []

        # Define convolutional blocks
        self.conv_blocks = []
        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = keras.layers.Input(shape=input_shape)
            self.input_layers.append(input_layer)
            
            conv1 = keras.layers.Conv1D(
                filters=int(self.filters_multipliers[channel_id] * 128),
                kernel_size=int(self.kernel_size_multipliers[channel_id] * 8),
                padding='same'
            )(input_layer)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation(activation='relu')(conv1)

            conv2 = keras.layers.Conv1D(
                filters=int(self.filters_multipliers[channel_id] * 256),
                kernel_size=int(self.kernel_size_multipliers[channel_id] * 5),
                padding='same'
            )(conv1)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)

            conv3 = keras.layers.Conv1D(
                filters=int(self.filters_multipliers[channel_id] * 128),
                kernel_size=int(self.kernel_size_multipliers[channel_id] * 3),
                padding='same'
            )(conv2)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)

            gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
            #self.conv_blocks.append(input_layer)
            self.conv_blocks.append(gap_layer)

        # Concatenate channel outputs
        if len(self.conv_blocks) > 1:
            merged_output = keras.layers.concatenate(self.conv_blocks, axis=-1)
        else:
            merged_output = self.conv_blocks[0]

        # Output layer
        output_layer = keras.layers.Dense(self.num_classes, activation='softmax')(merged_output)

        # Create model
        self.model = keras.models.Model(inputs=self.input_layers,  # Change here to use self.conv_blocks
                                        outputs=output_layer)

        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def call(self, inputs):
        return self.model(inputs)