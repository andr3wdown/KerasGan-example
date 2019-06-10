from keras.models import Sequential, Model, save_model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, BatchNormalization, Input, Reshape, Flatten, Deconv2D, UpSampling2D
import pickle
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import cv2
import time

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

__author__ = 'Chong Lee'

name = 'anime_gan'
disc_name = name + '_discriminator' + str(int(time.time())) + '_cnn'
gen_name = name + '_generator' + str(int(time.time())) + '_cnn'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

props = {
    'leak': 0.2,
    'lr': 0.0002,
    'drop': 0.3,
    'rho': 0.9,
    'momentum': 0.8,
    'disc_layers': [128, 64, 32],
    'gen_layers': [256, 256, 512, 512],
    'conv_gen': {
        'shape': (4, 4, 256)
    },
    'kernel': (3, 3),
    'pool': (2, 2)
}


def get_discriminator(input_shape):
    model = Sequential()

    for i in range(len(props['disc_layers'])):
        model.add(Conv2D(props['disc_layers'][i], props['kernel'], input_shape=input_shape))
        model.add(LeakyReLU(alpha=props['leak']))
        model.add(MaxPooling2D(props['pool']))
        model.add(Dropout(props['drop']))

    model.add(Flatten())

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=props['leak']))
    model.add(Dropout(props['drop']))

    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    image = Input(shape=input_shape)
    validity = model(image)
    return Model(image, validity)


def get_generator(noise_shape, output_shape):
    model = Sequential()

    model.add(Dense(props['gen_layers'][0], input_shape=noise_shape))
    model.add(BatchNormalization(momentum=props['momentum']))
    model.add(LeakyReLU(alpha=props['leak']))
    model.add(Dropout(props['drop']))

    for i in range(1, len(props['gen_layers'])):
        model.add(Dense(props['gen_layers'][i]))
        model.add(BatchNormalization(momentum=props['momentum']))
        model.add(LeakyReLU(alpha=props['leak']))
        model.add(Dropout(props['drop']))

    model.add(Dense(np.prod(output_shape), activation='sigmoid'))

    model.add(Reshape(output_shape))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)


def get_deconv_generator(noise_shape):
    model = Sequential()
    x, y, z = props['conv_gen']['shape']
    model.add(Dense(x * y * z, input_shape=noise_shape))
    model.add(BatchNormalization(momentum=props['momentum']))
    model.add(LeakyReLU(alpha=props['leak']))
    model.add(Dropout(props['drop']))

    model.add(Reshape(props['conv_gen']['shape']))

    for i in range(len(props['gen_layers'])):
        model.add(UpSampling2D())
        model.add(Conv2D(props['gen_layers'][i], kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=props['momentum']))
        model.add(LeakyReLU(alpha=props['leak']))

    model.add(Conv2D(3, kernel_size=3, padding="same", activation='sigmoid'))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)


def get_combo_model(image_shape, noise_shape, conv=True):
    opti = Adam(lr=props['lr'], beta_1=0.5, beta_2=0.5)#RMSprop(lr=props['lr'], rho=props['rho'])

    disc = get_discriminator(image_shape)
    disc.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy', 'mse'])

    gen = get_deconv_generator(noise_shape) if conv else get_generator(noise_shape, image_shape)
    gen.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy', 'mse'])

    z = Input(shape=noise_shape)
    img = gen(z)

    disc.trainable = False

    valid = disc(img)

    combo = Model(z, valid)
    combo.compile(loss='binary_crossentropy', optimizer=opti)
    return combo, disc, gen


def train_gan_model(dataset, noise_shape, batch_size=30, epochs=10000, verbose=0, conv=True):
    anime_dataset = pickle.load(open(dataset, 'rb'))
    anime_dataset = anime_dataset / 255

    shape_x, = noise_shape

    half_size = int(batch_size / 2)

    combo, disc, gen = get_combo_model(anime_dataset.shape[1:], noise_shape, conv=conv)

    for i in range(epochs):
        if i % 50 == 0:
            noise = np.random.uniform(0, 1.0, (half_size, shape_x))
            generated_images = gen.predict(noise)
            j = 0
            for image in generated_images:
                j += 1
                image = image * 255
                cv2.imwrite(f'generated/epoch{i}pict{j}.png', image)

        if verbose > 0 and verbose < 2:
            print(f'starting epoch {i + 1}')

#       discriminator
        idx = np.random.randint(0, anime_dataset.shape[0], half_size)
        images = anime_dataset[idx]

        noise = np.random.uniform(0, 1.0, (half_size, shape_x))

        generated_images = gen.predict(noise)

        disc_loss_real = disc.train_on_batch(images, np.ones((half_size, 1)))
        disc_loss_fake = disc.train_on_batch(generated_images, np.zeros((half_size, 1)))
        disc_loss = 0.5 * np.add(disc_loss_real, disc_loss_fake)

#       generator

        noise = np.random.uniform(0, 1.0, (batch_size, shape_x))
        valid_y = np.array([1] * batch_size)

        gen_loss = combo.train_on_batch(noise, valid_y)

        if verbose > 0 and verbose < 2:
            print(f'discriminator: {disc_loss} | generator: {gen_loss}')

    noise = np.random.uniform(0, 1.0, (half_size, shape_x))
    generated_images = gen.predict(noise)
    i = 0
    for image in generated_images:
        i += 1
        image = image*255
        cv2.imwrite(f'generated/{i}.png', image)


def load_classification_datasets():
    global X, y, test_X, test_y
    X = pickle.load(open('train_X.p', 'rb'))
    y = pickle.load(open('train_y.p', 'rb'))
    test_X = pickle.load(open('test_X.p', 'rb'))
    test_y = pickle.load(open('test_y.p', 'rb'))
    return X, y, test_X, test_y


if __name__ == '__main__':
    train_gan_model('datasets/anime_faces.p', (200, ), verbose=1, conv=False)



