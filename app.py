import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# ONLY USE CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("Current Working Directory:", os.getcwd())

class_order = ['Church', 'Mosque', 'Temple']
subclass_order = ['dravidian', 'nagara']
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 12

# -- Remove corrupt images --
def clean_images(folder):
    if not os.path.exists(folder):
        return
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            print(f"Removing corrupt image: {path}")
            try:
                os.remove(path)
            except Exception:
                pass

folders = [
    "train/Church", "train/Mosque", "train/Temple",
    "train/Temple/dravidian", "train/Temple/nagara",
    "test/Church", "test/Mosque", "test/Temple",
    "test/Temple/dravidian", "test/Temple/nagara"
]
for f in folders:
    clean_images(f)

# -- Print image counts --
def count_images(path):
    return len([f for f in os.listdir(path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
print("\nImage counts per class:")
for folder in ['train', 'test']:
    for c in class_order:
        p = f"{folder}/{c}"
        print(f"{p:25}: {count_images(p) if os.path.exists(p) else 0} images")
print("\nTemple subclass (style) counts:")
for folder in ['train/Temple/dravidian', 'train/Temple/nagara', 'test/Temple/dravidian', 'test/Temple/nagara']:
    print(f"{folder:32}: {count_images(folder) if os.path.exists(folder) else 0} images")

# -- Augmentation --
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=18,
    width_shift_range=0.14,
    height_shift_range=0.14,
    shear_range=0.13,
    zoom_range=0.13,
    brightness_range=[0.85, 1.15],
    horizontal_flip=True,
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

# -- First-Level Classifier --
train_flow = train_gen.flow_from_directory(
    'train',
    target_size=IMG_SIZE,
    classes=class_order,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_flow = val_gen.flow_from_directory(
    'test',
    target_size=IMG_SIZE,
    classes=class_order,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n=== Training First-Level Classifier ===")
base = tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, weights='imagenet')
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
output = tf.keras.layers.Dense(len(class_order), activation='softmax')(x)
model1 = tf.keras.Model(base.input, output)
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history1 = model1.fit(train_flow, epochs=EPOCHS, validation_data=val_flow)
model1.save('main_classifier.h5')
print(f"First-level train accuracy: {history1.history['accuracy'][-1]:.3f}")
print(f"First-level val accuracy:   {history1.history['val_accuracy'][-1]:.3f}")

# -- Second-Level Classifier --
train_temple = train_gen.flow_from_directory(
    'train/Temple',
    target_size=IMG_SIZE,
    classes=subclass_order,
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_temple = val_gen.flow_from_directory(
    'test/Temple',
    target_size=IMG_SIZE,
    classes=subclass_order,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\n=== Training Temple Style Classifier ===")
x2 = tf.keras.layers.GlobalAveragePooling2D()(base.output)
output2 = tf.keras.layers.Dense(len(subclass_order), activation='softmax')(x2)
model2 = tf.keras.Model(base.input, output2)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history2 = model2.fit(train_temple, epochs=EPOCHS, validation_data=val_temple)
model2.save('temple_classifier.h5')
print(f"Second-level train accuracy: {history2.history['accuracy'][-1]:.3f}")
print(f"Second-level val accuracy:   {history2.history['val_accuracy'][-1]:.3f}")

print("\nIf any validation accuracy is below ~0.7, check your folder images and class balance.")
print("Models saved as main_classifier.h5 and temple_classifier.h5.")
