import keras


def reshape_samples(samples):
    return [x.reshape((x.shape[0], 2, round(x.shape[1] / 2), 1)) for x in samples]


class MlpLstm(keras.Model):
    def __init__(self, input_shapes, nb_classes):
        super(MlpLstm, self).__init__()
        self.input_shapes = input_shapes
        self.nb_classes = nb_classes

        self.extra_dense_layers_no = 2
        self.dense_outputs = len(input_shapes) * [500]

        self.input_layers = []
        self.channel_outputs = []

        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = keras.layers.Input(shape=(None, round(input_shape[0] / 2), 1), name=f"input_for_{channel_id}")
            self.input_layers.append(input_layer)

            layer = keras.layers.TimeDistributed(keras.layers.Flatten())(input_layer)
            layer = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(layer)
            layer = keras.layers.TimeDistributed(keras.layers.Dense(self.dense_outputs[channel_id], activation='relu'))(layer)

            for _ in range(self.extra_dense_layers_no):
                layer = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(layer)
                layer = keras.layers.TimeDistributed(keras.layers.Dense(self.dense_outputs[channel_id], activation='relu'))(layer)

            layer = keras.layers.TimeDistributed(keras.layers.Dropout(0.3))(layer)
            layer = keras.layers.LSTM(self.dense_outputs[channel_id])(layer)
            self.channel_outputs.append(layer)

        if len(self.channel_outputs) > 1:
            merged_output = keras.layers.concatenate(self.channel_outputs, axis=-1)
        else:
            merged_output = self.channel_outputs[0]

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(merged_output)

        self.model = keras.models.Model(inputs=self.input_layers, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def call(self, inputs):
        inputs = reshape_samples(inputs)
        return self.model(inputs)
