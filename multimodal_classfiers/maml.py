from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam #from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense
import numpy as np
import time

from multimodal_classfiers.classifier import Classifier, get_multipliers

# Constants for MAML
META_TRAIN_ITERATIONS = 10
NUM_TASKS_PER_META_ITERATION = 5
META_EPOCHS = 5
TASK_SPECIFIC_EPOCHS = 5

class MAML(Classifier):
    def __init__(self, output_directory, input_shapes, nb_classes, verbose=False, hyperparameters=None, model_init=None):
        super().__init__(output_directory, input_shapes, nb_classes, verbose, hyperparameters, model_init)

        
    def build_model(self, input_shapes, nb_classes, hyperparameters):
        input_layers = []  
        channel_outputs = []

        filters_multipliers, kernel_size_multipliers = get_multipliers(len(input_shapes), hyperparameters)
        # print(f"{filters_multipliers=}")
        # print(f"{kernel_size_multipliers=}")
        
        for channel_id, input_shape in enumerate(input_shapes):
            input_layer = layers.Input(shape=input_shape)  # Define an input layer for each channel
            input_layers.append(input_layer)  # Store the input layer in the list

            # Convolutional layers for the current channel
            conv1 = Conv1D(filters=int(filters_multipliers[channel_id] * 128),
                          kernel_size=int(kernel_size_multipliers[channel_id] * 8), padding='same')(input_layer)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation(activation='relu')(conv1)

            conv2 = Conv1D(filters=int(filters_multipliers[channel_id] * 256),
                          kernel_size=int(kernel_size_multipliers[channel_id] * 5), padding='same')(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)

            conv3 = Conv1D(int(filters_multipliers[channel_id] * 128),
                          kernel_size=int(kernel_size_multipliers[channel_id] * 3), padding='same')(conv2)
            conv3 = BatchNormalization()(conv3)
            conv3 = Activation('relu')(conv3)

            gap_layer = GlobalAveragePooling1D()(conv3)  # Global average pooling to reduce spatial dimensions
            channel_outputs.append(gap_layer)  # Store the output of the current channel
            

        # Concatenate the output layers from all channels, if there are multiple channels
        flat = layers.concatenate(channel_outputs, axis=-1) if len(channel_outputs) > 1 else channel_outputs[0]

        # Final fully connected layer for classification
        output_layer = Dense(nb_classes, activation='softmax')(flat)

        # Create the model with multiple input channels and the final output layer
        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        # Compile the model with the specified loss, optimizer, and metrics
        model.compile(loss='categorical_crossentropy', optimizer=self.get_optimizer(), metrics=['accuracy'])

        return model
    
    
    def maml_train(self, tasks, batch_size, meta_epochs=10, task_specific_epochs=5, meta_iterations=10):
        # The 'tasks' parameter should be a list of tasks, each containing 'support' and 'query' data
        
        # Clone the initial model parameters for meta-updates
        initial_model_weights = self.model.get_weights()
        
        for meta_iteration in range(meta_iterations):
            print(f"Meta-Iteration {meta_iteration + 1} / {meta_iterations}")

            loss_for_all_tasks = []
            for task in tasks:
                support_set, query_set = task['support'], task['query'] 
                support_x, support_y = support_set['x'], support_set['y']   # 9, (444, 1, 100)
                query_x, query_y = query_set['x'], query_set['y']
                
                # Restore the initial model parameters
                self.model.set_weights(initial_model_weights)
                
                # Fine-tune the model on the support set for a few epochs
                self.model.fit(support_x, support_y, epochs=task_specific_epochs, batch_size=batch_size, verbose=self.verbose)

                # Evaluate the model on the query set for each task
                loss, _ = self.model.evaluate(query_x, query_y, verbose=self.verbose)
                print(f"Task Loss: {loss:.4f}, Task Accuracy: {accuracy:.4f}")
                loss_for_all_tasks.append(loss)
            
            # Calculate the gradient of meta-objective (e.g., loss across all tasks) with respect to initial_model_weights
            meta_objective = sum(loss_for_all_tasks)
            
            # Calculate the gradient of the meta-objective
            gradients = K.gradients(meta_objective, self.model.trainable_weights)
            
            # Update the initial_model_weights using the gradients
            updates = [initial_model_weight - learning_rate * gradient for initial_model_weight, gradient in zip(initial_model_weights, gradients)]
            self.model.set_weights(updates)
            
        print("Meta-Training Complete")
    
    # For customizing the evaluation method for your specific problem
    # maml_test
    def evaluate(self, x_test, y_true):
        # Ensure that the model is initialized
        if self.model is None:
            raise ValueError("The model must be initialized before evaluation.")

        # Load the best model weights
        self.model.load_weights(self.output_directory + 'best_model.hdf5')
        
        # Evaluate the model on the test data
        loss, y_pred_probabilities = self.model.evaluate(x_test, y_true, verbose=0)
        y_pred = np.argmax(y_pred_probabilities, axis=1)

        # Calculate additional evaluation metrics
        accuracy = accuracy_score(np.argmax(y_true, axis=1), y_pred)
        f1 = f1_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_pred_probabilities, average='weighted')
        precision = precision_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
        recall = recall_score(np.argmax(y_true, axis=1), y_pred, average='weighted')

        return loss, y_pred, y_pred_probabilities #, accuracy, f1, auc, precision, recall
    
    def get_optimizer(self):
        return Adam(learning_rate=self.hyperparameters.lr, decay=self.hyperparameters.decay)
