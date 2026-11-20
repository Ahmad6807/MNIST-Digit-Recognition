import tensorflow as tf
from typing import Tuple, Optional


def build_mnist_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1),
                    num_classes: int = 10) -> tf.keras.Model:
    """Build the CNN architecture specified in the PRD.

    Architecture:
        Input (28, 28, 1)
        Conv2D (32, 3x3, ReLU, same padding)
        Conv2D (64, 3x3, ReLU, same padding)
        MaxPooling2D (2x2)
        Dropout (0.25)
        Flatten
        Dense (128, ReLU)
        Dropout (0.5)
        Dense (10, Softmax)

    Total trainable params: ~1,625,866
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="conv1")(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="conv2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = tf.keras.layers.Dropout(0.25, name="dropout_conv")(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(128, activation='relu', name="dense1")(x)
    x = tf.keras.layers.Dropout(0.5, name="dropout_dense")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='MNIST_CNN')
    return model


def compile_model(model: tf.keras.Model,
                  learning_rate: float = 0.001) -> tf.keras.Model:
    """Compile model with categorical cross-entropy and Adam optimizer."""
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


def get_callbacks(config: dict) -> list:
    """Create training callbacks from config."""
    callbacks = []

    cbs = config.get("training", {}).get("callbacks", {})

    if "early_stopping" in cbs:
        es = cbs["early_stopping"]
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=es.get("monitor", "val_loss"),
            patience=es.get("patience", 5),
            restore_best_weights=es.get("restore_best_weights", True),
            verbose=1
        ))

    if "reduce_lr" in cbs:
        rl = cbs["reduce_lr"]
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=rl.get("monitor", "val_loss"),
            factor=rl.get("factor", 0.5),
            patience=rl.get("patience", 3),
            min_lr=rl.get("min_lr", 1e-6),
            verbose=1
        ))

    if "model_checkpoint" in cbs:
        mc = cbs["model_checkpoint"]
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=mc.get("filepath", "models/best_cnn_model.keras"),
            monitor=mc.get("monitor", "val_accuracy"),
            mode=mc.get("mode", "max"),
            save_best_only=mc.get("save_best_only", True),
            verbose=1
        ))

    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir="logs/fit",
        histogram_freq=0,
        write_graph=True
    ))

    return callbacks
