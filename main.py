import os
import cv2
import uuid
import time
import tensorflow as tf
import json
import numpy as np
import albumentations as alb
from matplotlib import pyplot as plt

# Function to collect images using OpenCV
def collect_images(number_images=30):
    IMAGES_PATH = os.path.join('data', 'images')
    os.makedirs(IMAGES_PATH, exist_ok=True)
    cap = cv2.VideoCapture(1)
    for imgnum in range(number_images):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(500)  # Wait for 0.5 seconds

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to annotate images using LabelMe
def annotate_images():
    !labelme

# Function to setup Albumentations transform pipeline
def setup_augmentation_pipeline():
    return alb.Compose([
        alb.RandomCrop(width=450, height=450),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.2),
        alb.RandomGamma(p=0.2),
        alb.RGBShift(p=0.2),
        alb.VerticalFlip(p=0.5)
    ], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))

# Function to apply augmentations and view results
def apply_augmentations(img, label, augmentor):
    coords = [label['shapes'][0]['points'][0][0], label['shapes'][0]['points'][0][1],
              label['shapes'][0]['points'][1][0], label['shapes'][0]['points'][1][1]]
    coords = list(np.divide(coords, [640, 480, 640, 480]))
    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
    plt.imshow(augmented['image'])
    plt.show()

# Function to save the model
def save_model(model, model_path='facetracker.h5'):
    model.save(model_path)

# Function for real-time detection
def real_time_detection(facetracker):
    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        _, frame = cap.read()
        frame = frame[50:500, 50:500, :]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            cv2.rectangle(frame,
                          tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                          (255, 0, 0), 2)
            cv2.rectangle(frame,
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [80, 0])),
                          (255, 0, 0), -1)
            cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('FaceTrack', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main function to execute the pipeline
def main():
    collect_images()
    annotate_images()
    augmentor = setup_augmentation_pipeline()
    # Apply augmentation to an example image (you can replace this with your data)
    img = cv2.imread('data/train/images/ffd85fc5-cc1a-11ec-bfb8-a0cec8d2d278.jpg')
    with open('data/train/labels/ffd85fc5-cc1a-11ec-bfb8-a0cec8d2d278.json', 'r') as f:
        label = json.load(f)
    apply_augmentations(img, label, augmentor)
    # Continue with the rest of the pipeline...

if __name__ == "__main__":
    main()
    # Load image into TF data pipeline
    def load_image(x):
        byte_img = tf.io.read_file(x)
        img = tf.io.decode_jpeg(byte_img)
        return img

    images = tf.data.Dataset.list_files('data\\images\\*.jpg')
    images = images.map(load_image)

    # View raw images with Matplotlib
    image_generator = images.batch(4).as_numpy_iterator()
    plot_images = image_generator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx, image in enumerate(plot_images):
        ax[idx].imshow(image)
    plt.show()

    # Partition unaugmented data
    for folder in ['train', 'test', 'val']:
        for file in os.listdir(os.path.join('data', folder, 'images')):
            filename = file.split('.')[0] + '.json'
            existing_filepath = os.path.join('data', 'labels', filename)
            if os.path.exists(existing_filepath):
                new_filepath = os.path.join('data', folder, 'labels', filename)
                os.replace(existing_filepath, new_filepath)

    # Apply image augmentation on images and labels using Albumentations
    def apply_augmentations():
        for partition in ['train', 'test', 'val']:
            for image in os.listdir(os.path.join('data', partition, 'images')):
                img = cv2.imread(os.path.join('data', partition, 'images', image))

                coords = [0, 0, 0.00001, 0.00001]
                label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        label = json.load(f)

                    coords[0] = label['shapes'][0]['points'][0][0]
                    coords[1] = label['shapes'][0]['points'][0][1]
                    coords[2] = label['shapes'][0]['points'][1][0]
                    coords[3] = label['shapes'][0]['points'][1][1]
                    coords = list(np.divide(coords, [640, 480, 640, 480]))

                try:
                    for x in range(60):
                        augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                        cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'),
                                    augmented['image'])

                        annotation = {}
                        annotation['image'] = image

                        if os.path.exists(label_path):
                            if len(augmented['bboxes']) == 0:
                                annotation['bbox'] = [0, 0, 0, 0]
                                annotation['class'] = 0
                            else:
                                annotation['bbox'] = augmented['bboxes'][0]
                                annotation['class'] = 1
                        else:
                            annotation['bbox'] = [0, 0, 0, 0]
                            annotation['class'] = 0

                        with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'),
                                  'w') as f:
                            json.dump(annotation, f)

                except Exception as e:
                    print(e)

    apply_augmentations()

    # Load augmented images to TensorFlow dataset
    def load_labels(label_path):
        with open(label_path.numpy(), 'r', encoding="utf-8") as f:
            label = json.load(f)
        return [label['class']], label['bbox']

    train_images = tf.data.Dataset.list_files('aug_data\\train\\images\\*.jpg', shuffle=False)
    train_images = train_images.map(load_image)
    train_images = train_images.map(lambda x: tf.image.resize(x, (120, 120)))
    train_images = train_images.map(lambda x: x / 255)

    train_labels = tf.data.Dataset.list_files('aug_data\\train\\labels\\*.json', shuffle=False)
    train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    test_images = tf.data.Dataset.list_files('aug_data\\test\\images\\*.jpg', shuffle=False)
    test_images = test_images.map(load_image)
    test_images = test_images.map(lambda x: tf.image.resize(x, (120, 120)))
    test_images = test_images.map(lambda x: x / 255)

    test_labels = tf.data.Dataset.list_files('aug_data\\test\\labels\\*.json', shuffle=False)
    test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    val_images = tf.data.Dataset.list_files('aug_data\\val\\images\\*.jpg', shuffle=False)
    val_images = val_images.map(load_image)
    val_images = val_images.map(lambda x: tf.image.resize(x, (120, 120)))
    val_images = val_images.map(lambda x: x / 255)

    val_labels = tf.data.Dataset.list_files('aug_data\\val\\labels\\*.json', shuffle=False)
    val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

    # Combine label and image samples
    train = tf.data.Dataset.zip((train_images, train_labels))
    train = train.shuffle(5000)
    train = train.batch(8)
    train = train.prefetch(4)

    test = tf.data.Dataset.zip((test_images, test_labels))
    test = test.shuffle(1300)
    test = test.batch(8)
    test = test.prefetch(4)

    val = tf.data.Dataset.zip((val_images, val_labels))
    val = val.shuffle(1000)
    val = val.batch(8)
    val = val.prefetch(4)

    # Define the model
    def build_model():
        input_layer = tf.keras.layers.Input(shape=(120, 120, 3))
        vgg = tf.keras.applications.VGG16(include_top=False)(input_layer)
        f1 = tf.keras.layers.GlobalMaxPooling2D()(vgg)
        class1 = tf.keras.layers.Dense(2048, activation='relu')(f1)
        class2 = tf.keras.layers.Dense(1, activation='sigmoid')(class1)
        f2 = tf.keras.layers.GlobalMaxPooling2D()(vgg)
        regress1 = tf.keras.layers.Dense(2048, activation='relu')(f2)
        regress2 = tf.keras.layers.Dense(4, activation='sigmoid')(regress1)
        facetracker = tf.keras.Model(inputs=input_layer, outputs=[class2, regress2])
        return facetracker

    facetracker = build_model()

    # Define losses and optimizers
    batches_per_epoch = len(train)
    lr_decay = (1. / 0.75 - 1) / batches_per_epoch
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=lr_decay)

    def localization_loss(y_true, yhat):
        delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] -
