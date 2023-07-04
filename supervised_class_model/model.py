import tensorflow as tf
from supervised_class_model.attention_layer import AttentionLayer


def build_att(window_size, filters, kernel_size, units):
    """Build an SGN NILM model with an attention layer in the regression subnetwork

    Args:
        window_size (_type_): _description_
        filters (_type_): _description_
        kernel_size (_type_): _description_
        units (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_data = tf.keras.Input(shape=(window_size, 1))

    # CLASSIFICATION SUBNETWORK
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu')(input_data)
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
    classification_output = tf.keras.layers.Dense(units=window_size, activation='sigmoid', name="classification_output")(x)

    #REGRESSION SUBNETWORK
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input_data)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, activation="tanh", return_sequences=True), merge_mode="concat")(y)
    y, weights = AttentionLayer(units=units)(y)
    y = tf.keras.layers.Dense(units, activation='relu')(y)
    regression_output = tf.keras.layers.Dense(window_size, activation='relu', name="regression_output")(y)

    output = tf.keras.layers.Multiply(name="output")([regression_output, classification_output])

    full_model = tf.keras.Model(inputs=input_data, outputs=[output, classification_output], name="LDwA")
    attention_model = tf.keras.Model(inputs=input_data, outputs=weights)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    full_model.compile(optimizer=optimizer, loss={
        "output": tf.keras.losses.MeanSquaredError(),
        "classification_output": tf.keras.losses.BinaryCrossentropy()})
    
    attention_model.compile(optimizer='sgd', loss='mse')

    return full_model, attention_model


def build_supervised_class_model(window_size, filters, kernel_size, units):
    """Builds a model to classify operational states of an appliance as ON/OFF from an aggregate load

    Args:
        window_size (_type_): _description_
        filters (_type_): _description_
        kernel_size (_type_): _description_
        units (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_data = tf.keras.Input(shape=(window_size, 1))

    # CLASSIFICATION SUBNETWORK
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu')(input_data)
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
    classification_output = tf.keras.layers.Dense(units=window_size, activation='sigmoid', name="classification_output")(x)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    s_class_model = tf.keras.Model(inputs=input_data, outputs=classification_output, name="supervised_class")

    s_class_model.compile(optimizer=optimizer, loss={
        "classification_output": tf.keras.losses.BinaryCrossentropy()})
    
    return s_class_model

def build_sgn(window_size, filters, kernel_size, units):
    """Build an SGN NILM model

    Args:
        window_size (_type_): _description_
        filters (_type_): _description_
        kernel_size (_type_): _description_
        units (_type_): _description_

    Returns:
        _type_: _description_
    """
    input_data = tf.keras.Input(shape=(window_size, 1))

    # CLASSIFICATION SUBNETWORK
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, activation='relu')(input_data)
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation='relu', kernel_initializer='he_normal')(x)
    classification_output = tf.keras.layers.Dense(units=window_size, activation='sigmoid', name="classification_output")(x)

    #REGRESSION SUBNETWORK
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input_data)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, activation="tanh"), merge_mode="concat")(y)
    #y, weights = AttentionLayer(units=units)(y)
    y = tf.keras.layers.Dense(units, activation='relu')(y)
    regression_output = tf.keras.layers.Dense(window_size, activation='relu', name="regression_output")(y)

    output = tf.keras.layers.Multiply(name="output")([regression_output, classification_output])

    full_model = tf.keras.Model(inputs=input_data, outputs=[output, classification_output], name="SGN")
    #attention_model = tf.keras.Model(inputs=input_data, outputs=weights)
    attention_model=None

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    full_model.compile(optimizer=optimizer, loss={
        "output": tf.keras.losses.MeanSquaredError(),
        "classification_output": tf.keras.losses.BinaryCrossentropy()})

    return full_model

