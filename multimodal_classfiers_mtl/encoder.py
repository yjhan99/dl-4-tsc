import keras
import keras_contrib

# from multimodal_classfiers.classifier import Classifier, get_multipliers
from multimodal_classfiers_finetuning.classifier_finetuning import Classifier, get_multipliers


class ClassifierEncoder(Classifier):
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layesrs = []
        channel_outputs = []

        filters_multipliers, kernel_size_multipliers = get_multipliers(len(input_shapes), hyperparameters)

        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = keras.layers.Input(input_shape)
            input_layesrs.append(input_layer)

            # conv block -1
            conv1 = keras.layers.Conv1D(filters=int(filters_multipliers[channel_id] * 128),
                                        kernel_size=int(kernel_size_multipliers[channel_id] * 5), strides=1,
                                        padding='same')(input_layer)
            conv1 = keras_contrib.layers.InstanceNormalization()(conv1)
            conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
            conv1 = keras.layers.Dropout(rate=0.2)(conv1)
            conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
            # conv block -2
            conv2 = keras.layers.Conv1D(filters=int(filters_multipliers[channel_id] * 256),
                                        kernel_size=int(kernel_size_multipliers[channel_id] * 11), strides=1,
                                        padding='same')(conv1)
            conv2 = keras_contrib.layers.InstanceNormalization()(conv2)
            conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
            conv2 = keras.layers.Dropout(rate=0.2)(conv2)
            conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
            # conv block -3
            conv3 = keras.layers.Conv1D(filters=int(filters_multipliers[channel_id] * 512),
                                        kernel_size=int(kernel_size_multipliers[channel_id] * 21), strides=1,
                                        padding='same')(conv2)
            conv3 = keras_contrib.layers.InstanceNormalization()(conv3)
            conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
            conv3 = keras.layers.Dropout(rate=0.2)(conv3)
            # split for attention
            attention_data = keras.layers.Lambda(lambda x: x[:, :, :int(filters_multipliers[channel_id] * 256)])(conv3)
            attention_softmax = keras.layers.Lambda(lambda x: x[:, :, int(filters_multipliers[channel_id] * 256):])(
                conv3)
            # attention mechanism
            attention_softmax = keras.layers.Softmax()(attention_softmax)
            multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
            # last layer
            dense_layer = keras.layers.Dense(units=int(filters_multipliers[channel_id] * 256), activation='sigmoid')(
                multiply_layer)
            dense_layer = keras_contrib.layers.InstanceNormalization()(dense_layer)
            # output layer
            flatten_layer = keras.layers.Flatten()(dense_layer)
            channel_outputs.append(flatten_layer)

        flatten_layer = keras.layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else \
            channel_outputs[0]
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layesrs, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model
