import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Loads in the dataset
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

#labels between 0 to 9 (stands for t-shirt, trousers, etc.)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Scaling down pixel values to make computations easier
train_images = train_images/255.0
test_images = test_images/255.0

#Displays images of clothes
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()

#model will determine what class(0-9) it thinks the image of clothing is
#each image of clothing is made up of 28x28 pixels(28 arrays of 28 pixels, with each array value being between 0 and 1 due to earlier scaling down code)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #flatten data into one array of 784 array to pass to neurons
    keras.layers.Dense(128, activation="relu"), #neurons are connected to others in hidden network
    keras.layers.Dense(10, activation="softmax") #output layer, activation softmax: will pick probabilities for each 0-9 value for a piece of clothing between 0 to 1
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#train model
#epoch: how many times the model will see this information(see the same image 5 times)
model.fit(train_images, train_labels, epochs=5)

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Tested Acc", test_acc)


prediction = model.predict(test_images) #all 10 images
#each prediction array value represents what model thinks is the probability that a certain 0-9 value is associated with an image

#Shows prediction of clothing article name & actual clothing article name
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("prediction" + prediction[i])
    plt.show()

#model gives name based on the highest probability of the 0-9 values to identify what clothing article an image is
#print(class_names[np.argmax(prediction[0])]) #if index is 0, t-shirt

