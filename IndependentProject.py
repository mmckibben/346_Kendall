import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = '/mnt/c/Users/kenda/Desktop/ecol 346/Independent Project/Images/Train'
validation_dir = '/mnt/c/Users/kenda/Desktop/ecol 346/Independent Project/Images/Valid'
test_dir = '/mnt/c/Users/kenda/Desktop/ecol 346/Independent Project/Images/Test'

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['Normal', 'Cancer'])

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['Normal', 'Cancer'])

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary',
        classes=['Normal', 'Cancer'])

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=40,
                    validation_data=validation_generator,
                    steps_per_epoch=10,
                    validation_steps=5)

test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)
model.save('goodmodel1.h5')

%matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

model = load_model('goodmodel1.h5')

image_paths = ['/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Cancer/Test.png', '/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Normal/Test2.png', '/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Cancer/Test3.png','/mnt/c/Users/kenda/OneDrive/Desktop/ecol 346/Independent Project/Images/Test/Normal/Test4.png']

def predict(model, image_paths):
    for i, img_path in enumerate(image_paths):
        # load the image
        img = load_img(img_path, target_size=(150, 150))
        
        # display the image
        plt.imshow(np.uint8(img))
        plt.show()

        # preprocess the image
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.

        # make prediction
        pred = model.predict(x)[0]
        
        # print the prediction result
        if pred <= 0.5:
            print(f"Image {i+1} is predicted as normal with confidence {100-pred.item()*100:.2f}%")
        else:
            print(f"Image {i+1} is predicted as cancer with confidence {pred.item()*100:.2f}%")
        print('\n')
        
        predict(model, image_paths)
