import config
import numpy as np
import pandas as pd
import tensorflow as tf
import utilities as utils
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50V2, ResNet101V2, ResNet152V2, InceptionResNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, ZeroPadding2D, Dropout


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet_with_depth_3(bb_name, input_shape, num_classes):
    K.clear_session()
    inception_resnet_toggle = utils.get_ir_toggle_status(bb_name)
    layer_list = utils.get_layer_list(bb_name)

    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained Model """
    encoder = utils.get_model(bb_name, inputs)

    """ Encoder """
    s1 = encoder.get_layer(layer_list[0]).output          ## (512 x 512)

    s2 = encoder.get_layer(layer_list[1]).output          ## (255 x 255) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)        ## (256 x 256)

    s3 = encoder.get_layer(layer_list[2]).output          ## (126 x 126) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s3 = ZeroPadding2D((1, 1))(s3)                    ## (128 x 128)

    """ Bridge """
    b1 = encoder.get_layer(layer_list[3]).output          ## (61 x 61) for 'InceptionResnet'
    if(inception_resnet_toggle):
        b1 = ZeroPadding2D(((2, 1), (2, 1)))(b1)          ## (64 x 64)

    """ Decoder """
    d1 = decoder_block(b1, s3, 256)                       ## (128 x 128)
    d2 = decoder_block(d1, s2, 128)                       ## (256 x 256)
    d3 = decoder_block(d2, s1, 64)                        ## (512 x 512)

    """ Output """
    dropout = Dropout(0.3)(d3)
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs)
    return model


def build_unet_with_depth_4(bb_name, input_shape, num_classes):
    K.clear_session()
    inception_resnet_toggle = utils.get_ir_toggle_status(bb_name)
    layer_list = utils.get_layer_list(bb_name)

    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained Model """
    encoder = utils.get_model(bb_name, inputs)

    """ Encoder """
    s1 = encoder.get_layer(layer_list[0]).output          ## (512 x 512)

    s2 = encoder.get_layer(layer_list[1]).output          ## (255 x 255) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)        ## (256 x 256)

    s3 = encoder.get_layer(layer_list[2]).output          ## (126 x 126) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s3 = ZeroPadding2D((1, 1))(s3)                    ## (128 x 128)

    s4 = encoder.get_layer(layer_list[3]).output          ## (61 x 61) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)         ## (64 x 64)

    """ Bridge """
    b1 = encoder.get_layer(layer_list[4]).output          ## (30 x 30)
    if(inception_resnet_toggle):
        b1 = ZeroPadding2D((1, 1))(b1)                    ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                       ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                       ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                       ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                        ## (512 x 512)

    """ Output """
    dropout = Dropout(0.3)(d4)
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs)
    return model


def build_unet_with_depth_5(bb_name, input_shape, num_classes):
    K.clear_session()
    inception_resnet_toggle = utils.get_ir_toggle_status(bb_name)
    layer_list = utils.get_layer_list(bb_name)

    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained Model """
    encoder = utils.get_model(bb_name, inputs)

    """ Encoder """
    s1 = encoder.get_layer(layer_list[0]).output          ## (512 x 512)

    s2 = encoder.get_layer(layer_list[1]).output          ## (255 x 255) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s2 = ZeroPadding2D(( (1, 0), (1, 0) ))(s2)        ## (256 x 256)

    s3 = encoder.get_layer(layer_list[2]).output          ## (126 x 126) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s3 = ZeroPadding2D((1, 1))(s3)                    ## (128 x 128)

    s4 = encoder.get_layer(layer_list[3]).output          ## (61 x 61) for 'InceptionResnet'
    if(inception_resnet_toggle):
        s4 = ZeroPadding2D(( (2, 1),(2, 1) ))(s4)         ## (64 x 64)

    s5 = encoder.get_layer(layer_list[4]).output          ## (30 x 30) for 'InceptionResnet'
    if(inception_resnet_toggle):
        b1 = ZeroPadding2D((1, 1))(s5)                    ## (32 x 32)

    """ Bridge """
    b1 = encoder.get_layer(layer_list[5]).output          ## (14 x 14) for 'InceptionResnet'
    if(inception_resnet_toggle):
        b1 = ZeroPadding2D((1, 1))(b1)                    ## (16 x 16)

    """ Decoder """
    d1 = decoder_block(b1, s5, 512)                       ## (32 x 32)
    d2 = decoder_block(d1, s4, 256)                       ## (64 x 64)
    d3 = decoder_block(d2, s3, 128)                       ## (128 x 128)
    d4 = decoder_block(d3, s2, 64)                        ## (256 x 256)
    d5 = decoder_block(d4, s1, 32)                        ## (512 x 512)

    """ Output """
    dropout = Dropout(0.3)(d5)
    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(dropout)

    model = Model(inputs, outputs)
    return model


def select_unet_depth(depth_selector, bb_name, input_shape, num_classes):
    if(depth_selector == '3'):
        return build_unet_with_depth_3(bb_name, input_shape, num_classes)
    if(depth_selector == '4'):
        return build_unet_with_depth_4(bb_name, input_shape, num_classes)
    if(depth_selector == '5'):
        return build_unet_with_depth_5(bb_name, input_shape, num_classes)


def build_model(depth_selector, bb_name, input_shape, num_classes, optimizer):
    model = select_unet_depth(depth_selector, bb_name, input_shape, num_classes)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=[utils.dice_coef, "accuracy"])
    model.summary()
    return model


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 **(epoch / s)
    return exponential_decay_fn


epochs = config.epoch_list[0]
scheduler_epoch = int(lr_scheduler_epoch_factor*epochs)
early_stopping_epoch = int(early_stopping_epoch_factor*epochs)


def scheduler(epoch, lr):
  if epoch < scheduler_epoch:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


def train_model(model, epochs, train_generator, validation_generator, steps_per_epoch, validation_steps, model_path, history_path, multiprocessing_toggle):
  #scheduler_epoch = int(config.lr_scheduler_epoch_factor*epochs)
  #early_stopping_epoch = int(config.early_stopping_epoch_factor*epochs)

  lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
  checkpoint = ModelCheckpoint(filepath = model_path, save_best_only = True, monitor = 'val_loss', mode = 'auto', verbose = 1)
  earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 12, mode = 'auto', verbose = 1, restore_best_weights = True)
  csvlogger = CSVLogger(filename = history_path, separator = ",", append = False)
  callbacks = [checkpoint, earlystop, csvlogger, lr_scheduler]

  # Training model
  hist = model.fit(train_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data = validation_generator,
                    validation_steps = validation_steps,
                    epochs = epochs,
                    callbacks=callbacks,
                    use_multiprocessing=multiprocessing_toggle,
                    verbose=1)

  return model, hist
