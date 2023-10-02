import keras

# from multimodal_classfiers.classifier import Classifier
from multimodal_classfiers_finetuning.classifier_finetuning import Classifier


class ClassifierMlp(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layers = []
        channel_outputs = []
        extra_dense_layers_no = 2
        dense_outputs = len(input_shapes) * [500]

        if hyperparameters is not None:
            extra_dense_layers_no = hyperparameters.extra_dense_layers_no
            dense_outputs = hyperparameters.dense_outputs

        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = keras.layers.Input(input_shape)
            input_layers.append(input_layer)

            # flatten/reshape because when multivariate all should be on the same axis
            input_layer_flattened = keras.layers.Flatten()(input_layer)

            layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
            layer = keras.layers.Dense(dense_outputs[channel_id], activation='relu')(layer_1)

            for i in range(extra_dense_layers_no):
                layer = keras.layers.Dropout(0.2)(layer)
                layer = keras.layers.Dense(dense_outputs[channel_id], activation='relu')(layer)

            output_layer = keras.layers.Dropout(0.3)(layer)
            channel_outputs.append(output_layer)

        flat = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(flat)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model
