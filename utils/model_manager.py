import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Dense, BatchNormalization, Activation,
    MaxPooling2D, Flatten
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1


class ModelManager:
    def __init__(self, model_type="cnn", input_shape=(28, 28, 1), n_classes=10):
        self.model_type = model_type.lower()
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build_model(self):
        if self.model_type == "cnn":
            return self._build_cnn_model()
        elif self.model_type == "mlp":
            return self._build_mlp_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _build_cnn_model(self):
        filters_per_conv_layer = [8, 8, 16]
        neurons_per_dense_layer = [32, 42]

        x_in = Input(self.input_shape)
        x = x_in

        for i, f in enumerate(filters_per_conv_layer):
            x = Conv2D(
                f,
                kernel_size=(3, 3),
                strides=(1, 1),
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
                use_bias=False,
                name=f'conv_{i}'
            )(x)
            x = BatchNormalization(name=f'bn_conv_{i}')(x)
            x = Activation('relu', name=f'conv_act_{i}')(x)
            x = MaxPooling2D(pool_size=(2, 2), name=f'pool_{i}')(x)

        x = Flatten()(x)

        for i, n in enumerate(neurons_per_dense_layer):
            x = Dense(
                n,
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
                use_bias=False,
                name=f'dense_{i}'
            )(x)
            x = BatchNormalization(name=f'bn_dense_{i}')(x)
            x = Activation('relu', name=f'dense_act_{i}')(x)

        x = Dense(self.n_classes, name='output_dense')(x)
        x_out = Activation('softmax', name='output_softmax')(x)

        return self._compile_model(x_in, x_out, name="cnn_model")


    def _build_mlp_model(self):
        neurons_per_layer = [128, 64]

        x_in = Input(self.input_shape)
        x = Flatten()(x_in)

        for i, n in enumerate(neurons_per_layer):
            x = Dense(
                n,
                kernel_initializer='lecun_uniform',
                kernel_regularizer=l1(0.0001),
                use_bias=False,
                name=f'dense_{i}'
            )(x)
            x = BatchNormalization(name=f'bn_dense_{i}')(x)
            x = Activation('relu', name=f'dense_act_{i}')(x)

        x = Dense(self.n_classes, name='output_dense')(x)
        x_out = Activation('softmax', name='output_softmax')(x)

        return self._compile_model(x_in, x_out, name="mlp_model")

    def _compile_model(self, x_in, x_out, name):
        model = Model(inputs=[x_in], outputs=[x_out], name=name)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=3e-3, beta_1=0.9, beta_2=0.999,
                epsilon=1e-07, amsgrad=True
            ),
            metrics=["accuracy"]
        )
        model.summary()
        return model
