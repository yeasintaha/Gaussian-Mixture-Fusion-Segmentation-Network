import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, UpSampling2D, concatenate, Layer
from keras.models import Model
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.optimizers import SGD, Adam


# Adding GMM segmentation to the input channels

HEIGHT = 256 
WIDTH = 256  

input1 = Input((HEIGHT, WIDTH, 3))
gmm_seg = Input((HEIGHT, WIDTH, 1))

# Encoder
conv1 = Conv2D(16, (3, 3), padding='same')(input1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
conv1 = Dropout(0.2)(conv1)

conv1 = concatenate([input1, gmm_seg, conv1], axis=-1)  # Updated concatenation
conv1 = Conv2D(16, (3, 3), padding='same')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Activation('relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

input2 = MaxPooling2D(pool_size=(2, 2))(input1)
gmm_seg2 = MaxPooling2D(pool_size=(2, 2))(gmm_seg)
conv21 = concatenate([input2, gmm_seg2, pool1], axis=-1)  # Updated concatenation

conv2 = Conv2D(32, (3, 3), padding='same')(conv21)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
conv2 = Dropout(0.2)(conv2)

conv2 = concatenate([conv21, conv2], axis=-1)
conv2 = Conv2D(32, (3, 3), padding='same')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Activation('relu')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

input3 = MaxPooling2D(pool_size=(2, 2))(input2)
gmm_seg3 = MaxPooling2D(pool_size=(2, 2))(gmm_seg2)
conv31 = concatenate([input3, gmm_seg3, pool2], axis=-1)  # Updated concatenation

conv3 = Conv2D(64, (3, 3), padding='same')(conv31)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
conv3 = Dropout(0.2)(conv3)

conv3 = concatenate([conv31, conv3], axis=-1)
conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Activation('relu')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

input4 = MaxPooling2D(pool_size=(2, 2))(input3)
gmm_seg4 = MaxPooling2D(pool_size=(2, 2))(gmm_seg3)
conv41 = concatenate([input4, gmm_seg4, pool3], axis=-1)  # Updated concatenation

conv4 = Conv2D(128, (3, 3), padding='same')(conv41)
conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)
conv4 = Dropout(0.2)(conv4)

conv4 = concatenate([conv41, conv4], axis=-1)
conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)
conv4 = Dropout(0.2)(conv4)

conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Activation('relu')(conv4)

# Decoder
conv5 = UpSampling2D(size=(2, 2))(conv4)
conv51 = concatenate([conv3, gmm_seg3, conv5], axis=-1)  # Updated concatenation

conv5 = Conv2D(64, (3, 3), padding='same')(conv51)
conv5 = BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)
conv5 = Dropout(0.2)(conv5)

conv5 = concatenate([conv51, conv5], axis=-1)
conv5 = Conv2D(64, (3, 3), padding='same')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = Activation('relu')(conv5)

conv6 = UpSampling2D(size=(2, 2))(conv5)
conv61 = concatenate([conv2, gmm_seg2, conv6], axis=-1)  # Updated concatenation

conv6 = Conv2D(32, (3, 3), padding='same')(conv61)
conv6 = BatchNormalization()(conv6)
conv6 = Activation('relu')(conv6)
conv6 = Dropout(0.2)(conv6)

conv6 = concatenate([conv61, conv6], axis=-1)
conv6 = Conv2D(32, (3, 3), padding='same')(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Activation('relu')(conv6)

conv7 = UpSampling2D(size=(2, 2))(conv6)
conv71 = concatenate([conv1, gmm_seg, conv7], axis=-1)  # Updated concatenation

conv7 = Conv2D(16, (3, 3), padding='same')(conv71)
conv7 = BatchNormalization()(conv7)
conv7 = Activation('relu')(conv7)
conv7 = Dropout(0.2)(conv7)

conv7 = concatenate([conv71, conv7], axis=-1)
conv7 = Conv2D(16, (3, 3), padding='same')(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Activation('relu')(conv7)

# Final
conv81 = UpSampling2D(size=(8, 8))(conv4)
conv82 = UpSampling2D(size=(4, 4))(conv5)
conv83 = UpSampling2D(size=(2, 2))(conv6)
conv8 = concatenate([conv81, conv82, conv83, conv7], axis=-1)
conv8 = Conv2D(1, (1, 1), activation='sigmoid')(conv8)
# conv8 = Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(conv8)

# Model definition
model = Model(inputs=[input1, gmm_seg], outputs=conv8)

print("Model compiling")

lr = 0.0001 
adm = Adam(lr)
model.compile(optimizer=adm, loss=binary_crossentropy, metrics=["accuracy"])

checkpoint_filepath = 'saved_model/model.weights.h5'

model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,  # Only save the weights
    monitor='val_loss',      # Monitor validation loss
    mode='min',              # Minimize validation loss
    save_best_only=True      # Save only the best model
)


print("Model training")
history = model.fit([train_images, train_fft_images], train_masks, validation_data=([val_images, val_fft_images], val_masks),  
          batch_size = 16, epochs=100, verbose=1, callbacks=[model_checkpoint_callback]) 
