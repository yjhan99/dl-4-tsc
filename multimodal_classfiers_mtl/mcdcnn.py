import keras

# from multimodal_classfiers.classifier import Classifier, get_multipliers
from multimodal_classfiers_finetuning.classifier_finetuning import Classifier, get_multipliers


class ClassifierMcdcnn(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        n_vars = len(input_shapes)
        padding = 'valid'

        filters_multipliers, kernel_size_multipliers = get_multipliers(len(input_shapes), hyperparameters)

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_shape = input_shapes[n_var]
            input_layer = keras.layers.Input(input_shape)
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=int(filters_multipliers[n_var] * 8),
                                              kernel_size=int(kernel_size_multipliers[n_var] * 5), activation='relu',
                                              padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=int(filters_multipliers[n_var] * 8),
                                              kernel_size=int(kernel_size_multipliers[n_var] * 5), activation='relu',
                                              padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers) if n_vars > 1 else conv2_layers[0]

        fully_connected = keras.layers.Dense(units=732, activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model
