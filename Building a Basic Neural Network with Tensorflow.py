import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

observations = 1000
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

generated_inputs = np.column_stack((xs, zs))
noise = np.random.uniform(-1, 1, (observations, 1))
generated_targets = 4 * xs - 6 * zs + 10 + noise

np.savez('basic_TF', inputs=generated_inputs, targets=generated_targets)
# I created a single .npz file to save the inputs and targets as numpy arrays.

training_data = np.load('basic_TF.npz')
input_size = 2
output_size = 1
# In this example, the output of my Regression model is 1.

# The 'Sequential' function specifies how my model will be created.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(output_size,
                          kernel_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1),
                          bias_initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
                          )
    # The 'Dense' function takes the provided inputs to the model and calculates the Dot Product of inputs and weights and adds the bias, for the operation xw +b.
])

custom_optimizer = tf.keras.optimizers.SGD(learning_rate=0.02)
# Here I chose the Stochastic Gradient Descent (SGD) as my optimizer because it is faster than the Gradient Descent.
# I expected the SGD to update the Weights many times in a single Epoch.
# However, SGD approximates, so I lost some accuracy using it, but this trade off was worth it.

model.compile(optimizer=custom_optimizer, loss='mean_squared_error')
# This function enabled me to select the optimizer and loss function of my model.
model.fit(training_data['inputs'], training_data['targets'], epochs=100, verbose=1)

model.layers[0].get_weights()
weights = model.layers[0].get_weights()[0]
weights
bias = model.layers[0].get_weights()[1]
bias
# I decided to save the weights and bias in separate variables for easier examination.

model.predict_on_batch(training_data['inputs'])
training_data['targets']

plt.plot(np.squeeze(model.predict_on_batch(training_data['inputs'])), np.squeeze(training_data['targets']))
# I squeezed the arrays to fit them into the standard of the plot function.
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()
