from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from dataset import dataset
from bag_and_bow import bag, bow
from predict import predict
from collections import deque

labels = [l['intent'] for l in dataset]

prepared_dataset = []

for d in [(bow(d['intent'], labels), [bag(v) for v in d['values']]) for d in dataset]:
    for v in d[1]:
        prepared_dataset.append(np.array([d[0], v], dtype=list))
        shuffle = v
        while (shuffle[-1] == 0):
            rotate = deque(shuffle)
            rotate.rotate(1)
            shuffle = list(rotate)
            prepared_dataset.append(np.array([d[0], shuffle], dtype=list))

pd_labels = [pd[0] for pd in prepared_dataset]
values = [pd[1] for pd in prepared_dataset]

print(len(pd_labels))
print(len(values))

model = Sequential()
model.add(Dense(255, input_shape=(255,), activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(.4))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(.4))
model.add(Dense(units=len(labels), activation='softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_x = np.asarray(list(values))
train_y = np.asarray(list(pd_labels)).reshape((-1, len(labels)))

# # train the model
model.fit(
    x=train_x,
    y=train_y,
    steps_per_epoch=len(train_x) // 16,
    batch_size=16,
    epochs=100,
    verbose=2,
)

prediction = predict(" Product of title", model, labels)

print(prediction)

model.save("model")
