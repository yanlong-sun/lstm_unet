import matplotlib.pyplot as plt
import heapq
import numpy as np


train_npy_path = "../lstm1/record/"

training_loss = np.load(train_npy_path + "train_loss.npy")
training_acc = np.load(train_npy_path + "train_acc.npy")
training_dice = np.load(train_npy_path + "train_dice.npy")
valid_loss = np.load(train_npy_path + "valid_loss.npy")
valid_acc = np.load(train_npy_path + "valid_acc.npy")
valid_dice = np.load(train_npy_path + "valid_dice.npy")


epoch = np.arange(1, 30001, 1000)

plt.plot(epoch, training_loss)
plt.plot(epoch, valid_loss)
plt.legend(['train', 'valid'], loc='upper right')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(epoch, training_acc)
plt.plot(epoch, valid_acc)
plt.legend(['train', 'valid'], loc='upper right')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(epoch, training_dice)
plt.plot(epoch, valid_dice)
plt.legend(['train', 'valid'], loc='upper right')
plt.ylabel('dice')
plt.xlabel('epoch')
plt.show()

print("best acc epoch(valid): " + str(heapq.nlargest(5, range(len(valid_acc)), valid_acc.take)))
print("best dice epoch(valid): " + str(heapq.nlargest(5, range(len(valid_dice)), valid_dice.take)))

print("best acc epoch(train): " + str(heapq.nlargest(5, range(len(training_acc)), training_acc.take)))
print("best dice epoch(train): " + str(heapq.nlargest(5, range(len(training_dice)), training_dice.take)))