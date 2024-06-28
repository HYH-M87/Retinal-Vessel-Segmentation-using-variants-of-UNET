import os
import cv2
import numpy as np
import skimage.io
# from patchify import patchify, unpatchify
from model import unetmodel,residual_attentionunet,attentionunet,residualunet
from tensorflow.keras.optimizers import Adam

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# CLAHE
def clahe_equalized(imgs):    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized

def patchify(image, patch_size, step):
    patches = []
    for i in range(0, image.shape[0] - patch_size + 1, step):
        for j in range(0, image.shape[1] - patch_size + 1, step):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
    patches = np.array(patches)
    return patches.reshape((image.shape[0] // patch_size, image.shape[1] // patch_size, patch_size, patch_size))

def unpatchify(patches):
    # print(patches.shape)
    image_shape = ( patches.shape[0] * patches.shape[2], patches.shape[1] * patches.shape[3])
    reconstructed_image = np.zeros(image_shape)
    patch_size = patches.shape[2]
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            reconstructed_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[i, j]
    return reconstructed_image

def main():

    patch_size = 512
    IMG_HEIGHT = patch_size
    IMG_WIDTH = patch_size
    IMG_CHANNELS = 1
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

    model = residual_attentionunet(input_shape)
    model.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights('Trained models/retina_AttentionRESUnet_150epochs.hdf5') # 加载权重
    
    img_dir='/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_ex/VOC2012/JPEGImages'
    mask_dir = '/home/hyh/Documents/quanyi/project/Data/e_optha_MA/MA_mask'
    img_files = os.listdir(img_dir)
    for i in img_files:
        input_image_path = os.path.join(img_dir, i)
        output_image_path = os.path.join('output_r', i)


        test_img = skimage.io.imread(input_image_path)
        mask_img = skimage.io.imread(os.path.join(mask_dir, i))

        test = test_img[:, :, 1]  
        test = clahe_equalized(test)

        # SIZE_X = (test_img.shape[1] // patch_size) * patch_size
        # SIZE_Y = (test_img.shape[0] // patch_size) * patch_size
        # test = cv2.resize(test, (SIZE_X, SIZE_Y))
        
        pad_height = (patch_size - test.shape[0] % patch_size) % patch_size
        pad_width = (patch_size - test.shape[1] % patch_size) % patch_size
        test = np.pad(test, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        mask_img = np.pad(mask_img, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
        test = np.array(test)

        

        # Pad the image with zeros
        
        
        # 创建图像块
        patches = patchify(test, patch_size, step=patch_size)
        predicted_patches = []

        # 模型预测每个图像块
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch = patches[i, j, :, :]
                single_patch_norm = (single_patch.astype('float32')) / 255.
                single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
                single_patch_input = np.expand_dims(single_patch_norm, 0)
                single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
                predicted_patches.append(single_patch_prediction)

        # 重构预测图像
        predicted_patches = np.array(predicted_patches)
        predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size, patch_size))
        reconstructed_image = unpatchify(predicted_patches_reshaped) * 255
        reconstructed_image = reconstructed_image * mask_img
        # 保存预测图像

        cv2.imwrite(output_image_path, reconstructed_image)

        print(f'预测图像已保存到 {output_image_path}')
    
if __name__ == "__main__":
    main()