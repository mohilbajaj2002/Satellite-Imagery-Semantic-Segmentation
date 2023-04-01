import config
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_history(history, model_name, save_mode=True):
    fig, ax = plt.subplots(1, 4, figsize=(40, 5))
    ax = ax.ravel()
    metrics = ['Dice Coefficient', 'Accuracy', 'Loss', 'Learning Rate']

    for i, met in enumerate(['dice_coef', 'accuracy', 'loss', 'lr']):
        if met != 'lr':
            ax[i].plot(history.history[met])
            ax[i].plot(history.history['val_' + met])
            ax[i].set_title('{} vs Epochs'.format(metrics[i]), fontsize=16)
            ax[i].set_xlabel('Epochs')
            ax[i].set_ylabel(metrics[i])
            ax[i].set_xticks(np.arange(0,46,4))
            ax[i].legend(['Train', 'Validation'])
            ax[i].xaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
            ax[i].yaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
        else:
            ax[i].plot(history.history[met])
            ax[i].set_title('{} vs Epochs'.format(metrics[i]), fontsize=16)
            ax[i].set_xlabel('Epochs')
            ax[i].set_ylabel(metrics[i])
            ax[i].set_xticks(np.arange(0,46,4))
            ax[i].xaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
            ax[i].yaxis.grid(True, color = "lightgray", linewidth = "0.8", linestyle = "-")
    if(save_mode):
        path = os.path.join(config.saved_history_plots_path, model_name + '.png')
        plt.savefig(path, facecolor= 'w',transparent= False, bbox_inches= 'tight', dpi= 150)
    else:
        plt.show()


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def display_performance_metrics(history, model_name):
    df = pd.DataFrame(history.history)
    val_loss_list = df.val_loss.values
    dice_coeff_list = df.val_dice_coef.values
    val_argmin = np.argmin(val_loss_list)
    dice = dice_coeff_list[val_argmin]
    print(f'Results for Model: {model_name}')
    print(f'Dice Coefficient: {round(dice, 3)}')
    print('')


def get_optimizer(opt_name, learning_rate):
    if(opt_name == 'SGD'):
        return tf.keras.optimizers.SGD(learning_rate)
    elif(opt_name == 'RMSProp'):
        return tf.keras.optimizers.RMSprop(learning_rate)
    elif(opt_name == 'Adam'):
        return tf.keras.optimizers.Adam(learning_rate)


# ['Resnet50', 'Resnet101', 'Resnet152', 'InceptionResNet']
def get_model(bb_name, inputs):
    if(bb_name == 'Resnet50'):
        return ResNet50V2(include_top=False, weights="imagenet", input_tensor=inputs)
    elif(bb_name == 'Resnet101'):
        return ResNet101V2(include_top=False, weights="imagenet", input_tensor=inputs)
    elif(bb_name == 'Resnet152'):
        return ResNet152V2(include_top=False, weights="imagenet", input_tensor=inputs)
    elif(bb_name == 'InceptionResNet'):
        return InceptionResNetV2(include_top=False, weights="imagenet", input_tensor=inputs)


def get_ir_toggle_status(bb_name):
    if(bb_name == 'InceptionResNet'):
        return True
    else:
        return False


def get_layer_list(bb_name):
    if(bb_name == 'Resnet50'):
        return ["input_1", "conv1_conv", "conv2_block3_1_relu", "conv3_block4_1_relu", "conv4_block6_1_relu", "conv5_block3_2_relu"]
    elif(bb_name == 'Resnet101'):
        return ["input_1", "conv1_conv", "conv2_block3_1_relu", "conv3_block4_1_relu", "conv4_block23_1_relu", "conv5_block3_2_relu"]
    elif(bb_name == 'Resnet152'):
        return ["input_1", "conv1_conv", "conv2_block3_1_relu", "conv3_block8_1_relu", "conv4_block36_1_relu", "conv5_block3_2_relu"]
    elif(bb_name == 'InceptionResNet'):
        return ["input_1", "activation", "activation_3", "activation_74", "activation_161", "activation_202"]


def get_input_shape(bb_name, input_shape):
    if(bb_name == 'Resnet152'):
        return (input_shape[0], input_shape[1])
    else:
        return (input_shape[2], input_shape[3])
