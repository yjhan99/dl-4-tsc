import keras

# from multimodal_classfiers.classifier import get_multipliers, Classifier
from multimodal_classfiers_mtl.classifier_mtl import get_multipliers, Classifier


class ClassifierFcn(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layers = []
        channel_outputs = []

        filters_multipliers, kernel_size_multipliers = get_multipliers(len(input_shapes), hyperparameters)

        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = keras.layers.Input(input_shape)
            input_layers.append(input_layer)

            conv1 = keras.layers.Conv1D(filters=int(filters_multipliers[channel_id] * 128),
                                        kernel_size=int(kernel_size_multipliers[channel_id] * 8), padding='same')(
                input_layer)
            conv1 = keras.layers.BatchNormalization()(conv1)
            conv1 = keras.layers.Activation(activation='relu')(conv1)

            conv2 = keras.layers.Conv1D(filters=int(filters_multipliers[channel_id] * 256),
                                        kernel_size=int(kernel_size_multipliers[channel_id] * 5), padding='same')(
                conv1)
            conv2 = keras.layers.BatchNormalization()(conv2)
            conv2 = keras.layers.Activation('relu')(conv2)

            conv3 = keras.layers.Conv1D(int(filters_multipliers[channel_id] * 128),
                                        kernel_size=int(kernel_size_multipliers[channel_id] * 3), padding='same')(conv2)
            conv3 = keras.layers.BatchNormalization()(conv3)
            conv3 = keras.layers.Activation('relu')(conv3)

            gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)
            channel_outputs.append(gap_layer)

        flat = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]

        # New dense layer added before the output
        additional_dense = keras.layers.Dense(128, activation='relu')(flat)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(additional_dense)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model
