import numpy as np
import random
import keras
# import matplotlib.pyplot as plt
from keras import backend as K
from keras.losses import mse
import warnings
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
import matplotlib.pyplot as plt
from joblib import dump, load


def save_model(model, input_shape, latent_dim, autoencoder_kwargs, path="", verbose=False):
    model_kwargs = {"input_shape": input_shape, "latent_dim": latent_dim, "autoencoder_kwargs": autoencoder_kwargs}
    model.save_weights(path + ".h5")
    dump(model_kwargs, path + ".joblib")
    return


def load_model(path, verbose=False):
    model_kwargs = load(path + ".joblib")
    encoder, decoder, autoencoder = build_vae(
        model_kwargs.get("input_shape"),
        model_kwargs.get("latent_dim"),
        model_kwargs.get("autoencoder_kwargs"),
        verbose=verbose
    )
    autoencoder.load_weights(path + ".h5")
    return encoder, decoder, autoencoder


def build_block(
        previous_layer,
        kind,
        filters=None,
        kernel_size=None,
        padding=None,
        activation=None,
        pooling=None,
        units=None,
        batch_normalization=None
):
    if kind == "encoder" or kind == "discriminator_gan":
        layer = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding)(previous_layer)
        if batch_normalization is not None:
            layer = keras.layers.normalization.BatchNormalization()(layer)
        if activation == "leakyrelu":
            layer = keras.layers.LeakyReLU()(layer)
        else:
            layer = keras.layers.Activation(activation)(layer)
        layer = keras.layers.MaxPooling1D(pooling)(layer)
    elif kind == "decoder" or kind == "generator":
        layer = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding)(previous_layer)
        if batch_normalization is not None:
            layer = keras.layers.normalization.BatchNormalization()(layer)
        if activation == "leakyrelu":
            layer = keras.layers.LeakyReLU()(layer)
        else:
            layer = keras.layers.Activation(activation)(layer)
        layer = keras.layers.UpSampling1D(size=pooling)(layer)
    elif kind == "discriminator":
        layer = keras.layers.Dense(units)(previous_layer)
        layer = keras.layers.Activation(activation)(layer)
    else:
        raise Exception("Block kind not valid")
    return layer


def build_residual_block(
        previous_layer,
        kind,
        n_layers,
        filters=None,
        kernel_size=None,
        padding=None,
        activation=None,
        pooling=None,
        units=None,
        batch_normalization=None
):
    if kind == "decoder":
        previous_layer = keras.layers.UpSampling1D(size=pooling)(previous_layer)
    input_layer = previous_layer
    for i in range(n_layers):
        block = build_block(
            previous_layer=previous_layer,
            kind=kind,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            pooling=1,
            units=units,
            batch_normalization=batch_normalization,
        )
        previous_layer = block
    shortcut = keras.layers.Conv1D(filters=filters, kernel_size=1, padding='same')(input_layer)
    if batch_normalization is not None:
        shortcut = keras.layers.normalization.BatchNormalization()(shortcut)
    output_layer = keras.layers.add([shortcut, block])
    if activation == "leakyrelu":
        output_layer = keras.layers.LeakyReLU()(output_layer)
    else:
        output_layer = keras.layers.Activation(activation)(output_layer)
    if kind == "encoder":
        output_layer = keras.layers.MaxPooling1D(pooling)(output_layer)
    return output_layer


def repeat_block(previous_layer,
                 kind,
                 n_layers,
                 filters=None,
                 kernel_size=None,
                 padding=None,
                 activation=None,
                 pooling=None,
                 units=None,
                 batch_normalization=None,
                 n_layers_residual=None
                 ):
    for i in range(n_layers):
        if n_layers_residual is None:
            block = build_block(
                previous_layer=previous_layer,
                kind=kind,
                filters=filters[i] if filters is not None else None,
                kernel_size=kernel_size[i] if kernel_size is not None else None,
                padding=padding[i] if padding is not None else None,
                activation=activation[i] if activation is not None else None,
                pooling=pooling[i] if pooling is not None else None,
                units=units[i] if units is not None else None,
                batch_normalization=batch_normalization
            )
            previous_layer = block
        else:
            block = build_residual_block(
                previous_layer=previous_layer,
                kind=kind,
                n_layers=n_layers_residual,
                filters=filters[i] if filters is not None else None,
                kernel_size=kernel_size[i] if kernel_size is not None else None,
                padding=padding[i] if padding is not None else None,
                activation=activation[i] if activation is not None else None,
                pooling=pooling[i] if pooling is not None else None,
                units=units[i] if units is not None else None,
                batch_normalization=batch_normalization
            )
            previous_layer = block
    return block


def plot_history(history_dict):
    plt.title("LOSS")
    plt.plot(history_dict["loss"], label="train")
    plt.plot(history_dict["val_loss"], label="validation")
    plt.legend()
    plt.show()

    if history_dict.get("mean_squared_error") is not None:
        plt.title("MSE")
        plt.plot(history_dict["mean_squared_error"], label="train")
        plt.plot(history_dict["val_mean_squared_error"], label="validation")
        plt.legend()
        plt.show()
    else:  # for newer versions of Keras
        plt.title("MSE")
        plt.plot(history_dict["mse"], label="train")
        plt.plot(history_dict["val_mse"], label="validation")
        plt.legend()
        plt.show()


def rec_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true, y_pred)  # * input_shape[0]


def rec_loss_sum(y_true, y_pred):
    return K.sum(keras.losses.mean_squared_error(y_true, y_pred), axis=-1)


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


def build_encoder(
        input_shape,
        latent_dim,
        **kwargs
):
    encoder_input = keras.layers.Input(shape=(input_shape))
    encoder_layers = repeat_block(
        encoder_input,
        "encoder",
        filters=kwargs.get("filters"),
        kernel_size=kwargs.get("kernel_size"),
        padding=kwargs.get("padding"),
        activation=kwargs.get("activation"),
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling"),
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual")
    )
    encoder_layers = keras.layers.Conv1D(
        filters=input_shape[1],  # FIXME: or 1?
        kernel_size=1,  # FIXME: maybe different value?
        padding="same")(encoder_layers)
    encoder_layers = keras.layers.Flatten()(encoder_layers)
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(encoder_layers)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(encoder_layers)
    z_mean, z_log_var = KLDivergenceLayer()([z_mean, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(.5 * t))(z_log_var)
    eps = Input(tensor=K.random_normal(stddev=1, shape=(K.shape(encoder_input)[0], latent_dim)))
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mean, z_eps])
    encoder_output = z
    encoder_input = [encoder_input, eps]
    encoder = keras.models.Model(encoder_input, encoder_output, name="Encoder")
    return encoder  # , z_mean, z_log_var


def build_decoder(
        encoder,
        latent_dim,
        **kwargs
):
    decoder_input = keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    decoder_layers = decoder_input
    decoder_layers = keras.layers.Dense(encoder.layers[-8].output_shape[1])(decoder_layers)
    decoder_layers = keras.layers.Reshape(encoder.layers[-9].output_shape[1:])(decoder_layers)

    decoder_layers = repeat_block(
        decoder_layers,
        "decoder",
        filters=kwargs.get("filters")[::-1],
        kernel_size=kwargs.get("kernel_size")[::-1],
        padding=kwargs.get("padding")[::-1],
        activation=kwargs.get("activation")[::-1],
        n_layers=kwargs.get("n_layers"),
        pooling=kwargs.get("pooling")[::-1],
        batch_normalization=kwargs.get("batch_normalization"),
        n_layers_residual=kwargs.get("n_layers_residual")
    )
    decoder_output = keras.layers.Conv1D(
        filters=encoder.input_shape[0][2],
        kernel_size=1,
        padding="same")(decoder_layers)
    decoder = keras.models.Model(decoder_input, decoder_output, name="Decoder")
    return decoder


def build_vae(
        input_shape,
        latent_dim,
        autoencoder_kwargs,
        verbose=True,
):
    encoder = build_encoder(
        input_shape,
        latent_dim,
        **autoencoder_kwargs
    )

    decoder = build_decoder(
        encoder,
        latent_dim,
        **autoencoder_kwargs
    )

    model_input = keras.layers.Input(shape=input_shape)
    eps = Input(tensor=K.random_normal(stddev=1, shape=(K.shape(model_input)[0], latent_dim)))
    model_input = [model_input, eps]
    model_output = decoder(encoder(model_input))
    autoencoder = keras.models.Model(model_input, model_output, name="VAE")

    autoencoder.compile(
        optimizer=autoencoder_kwargs.get("optimizer", keras.optimizers.Adam()),
        loss=autoencoder_kwargs.get("loss", rec_loss),
        metrics=["mse"])

    if verbose:
        encoder.summary()
        decoder.summary()
        autoencoder.summary()

    return encoder, decoder, autoencoder


if __name__ == "__main__":
    from datasets import build_cbf, build_ts_syege
    from blackbox_wrapper import BlackboxWrapper
    from joblib import load
    from utils import reconstruction_accuracy_vae, plot_reconstruction_vae
    from keras.utils.vis_utils import plot_model

    K.clear_session()

    (X_train, y_train, X_val, y_val,
     X_test, y_test, X_exp_train, y_exp_train,
     X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(n_samples=600)

    input_shape = X_train.shape[1:]
    latent_dim = 2

    autoencoder_kwargs = {
        "filters": [4, 4, 4, 4],
        "kernel_size": [3, 3, 3, 3],
        "padding": ["same", "same", "same", "same"],
        "activation": ["relu", "relu", "relu", "relu"],
        "pooling": [1, 1, 1, 1],
        "n_layers": 4,
        "optimizer": "adam",
        "n_layers_residual": None,
        "batch_normalization": False,
        "loss": rec_loss_sum
    }

    encoder, decoder, autoencoder = build_vae(input_shape, latent_dim, autoencoder_kwargs)
    autoencoder.fit(X_exp_train, X_exp_train, epochs=2000, validation_data=(X_exp_val, X_exp_val))
