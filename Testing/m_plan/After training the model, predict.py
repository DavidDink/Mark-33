# Create first network with Keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.utils import np_utils
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("w_victory.csv", delimiter=",")
# split into input (X) and output (Y) variables
Input = dataset[:, 0:8]

output = Dense
temp_Output = dataset[:, 9]
humd_Output = dataset[:, 11]


"""Output_array = ([dataset[:, 9], dataset[:, 11]])"""


"""
def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = numpy.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))
    
output_ohe = one_hot_encode_object_array(Output_array)
"""

#In order to have two outputs, we need to econde the output variable


# create model
model = Sequential()
model = Model(inputs=Input, outputs=[temp_Output, humd_Output])
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(9, init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(Input, [temp_Output, humd_Output], epochs=150, batch_size=10, verbose=2)
# evaluate the model
scores = model.evaluate(temp_Output, output_ohe)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# calculate predictions
predictions = model.predict(Input)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

