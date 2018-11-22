# Make sure set in appropriate working directory

# Make sure tensorflow and keras installed

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier  = Sequential()

# Convolution
classifier.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', input_shape = (32, 32, 1)))

# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding more convolutional layers and max pooling
classifier.add(Conv2D(filters = 32, kernel_size = (2, 2), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 46, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Creating Trainig and Test sets
training_set = train_datagen.flow_from_directory('Train',
                                                 target_size = (32, 32),
                                                 batch_size = 32,
                                                 color_mode = 'grayscale',
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('Test',
                                            target_size = (32, 32),
                                            batch_size = 32,
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical')

# Training the classifier
classifier.fit_generator(training_set,
                         steps_per_epoch = 2300,
                         epochs = 34,
                         validation_data = test_set,
                         validation_steps = 6900)