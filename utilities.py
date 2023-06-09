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


def make_prediction(model, prediction_gen, prediction_plot_path):
  count = 0
  for i in range(2):
      batch_img, batch_mask = next(prediction_gen)
      pred_all= model.predict(batch_img)
      np.shape(pred_all)

      for j in range(0,np.shape(pred_all)[0]):
          count += 1
          fig = plt.figure(figsize=(20,8))

          ax1 = fig.add_subplot(1,2,1)
          ax1.imshow(batch_img[j])
          ax1.set_title('Input Image', fontdict={'fontsize': 16, 'fontweight': 'medium'})
          ax1.grid(False)

          #ax2 = fig.add_subplot(1,3,2)
          #ax2.set_title('Ground Truth Mask', fontdict={'fontsize': 16, 'fontweight': 'medium'})
          #ax2.imshow(onehot_to_rgb(batch_mask[j],id2code))
          #ax2.grid(False)

          ax3 = fig.add_subplot(1,2,2)
          ax3.set_title('Predicted Mask', fontdict={'fontsize': 16, 'fontweight': 'medium'})
          ax3.imshow(onehot_to_rgb(pred_all[j],id2code))
          ax3.grid(False)

          plt.savefig(os.path.join(prediction_plot_path, f'prediction_{count}.png'), facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 200)
          plt.show()


def raw_data_viz(original_image_path, original_mask_path, misc_data_exploration_path):
    image = cv2.imread(original_image_path)
    mask = cv2.imread(original_mask_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

    fontsize = 16
    f, ax = plt.subplots(2, 1, figsize=(16, 12), squeeze=True)
    plt.tight_layout(w_pad=5, h_pad=5)

    ax[0].imshow(original_image)
    ax[0].set_title('Original Image', fontsize=fontsize)

    ax[1].imshow(original_mask)
    ax[1].set_title('Original Mask', fontsize=fontsize)

    filename = original_image_path.split('/')[-1]
    file_head = filename.split('.')[0]
    plt.savefig(os.path.join(misc_data_exploration_path, f'explore_raw_data_{file_head}.png'), facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)
    plt.show()
    

def raw_vs_transformed_data_viz(transformed_image_path, transformed_mask_path, original_image_path, original_mask_path, misc_data_exploration_path):

    original_image = cv2.imread(original_image_path)
    original_mask = cv2.imread(original_mask_path)
    transformed_image = cv2.imread(transformed_image_path)
    transformed_mask = cv2.imread(transformed_mask_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2RGB)
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    transformed_mask = cv2.cvtColor(transformed_mask, cv2.COLOR_BGR2RGB)

    fontsize = 16

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(10, 10), squeeze=True)
        f.set_tight_layout(h_pad=5, w_pad=5)

        ax[0].imshow(transformed_image)
        ax[1].imshow(transformed_mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(16, 12), squeeze=True)
        plt.tight_layout(pad=0.2, w_pad=1.0, h_pad=0.01)

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original Image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original Mask', fontsize=fontsize)

        ax[0, 1].imshow(transformed_image)
        ax[0, 1].set_title('Transformed Image', fontsize=fontsize)

        ax[1, 1].imshow(transformed_mask)
        ax[1, 1].set_title('Transformed Mask', fontsize=fontsize)

    filename = original_image_path.split('/')[-1]
    file_head = filename.split('.')[0]
    plt.savefig(os.path.join(misc_data_exploration_path, f'explore_raw_vs_transformed_data_{file_head}.png'), facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)
    plt.show()
