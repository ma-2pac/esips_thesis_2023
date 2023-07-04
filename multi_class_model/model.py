import tensorflow as tf

def build_multi_model(window_size, num_classes, kernel_size, units):

    #define sequential model
    model = tf.keras.Sequential()

    #add first block of 1D conv layers
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(window_size, 1)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # Second set of 1D convolutions and global max pooling
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPooling1D())

    # Dense layers
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))

    # Output layer
    model.add(tf.keras.layers.Dense(units=1, activation='softmax'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy','mse'])
    

    return model


