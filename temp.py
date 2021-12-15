import numpy as np
import pickle

pickle_in = open('trained_model.pkl', 'rb')
loaded_model = pickle.load(pickle_in)

input_data = (5,166,72,19,175,25.8,0.587,51)
modified = np.asarray(input_data)
reshaped_data = modified.reshape(1, -1)




predition = loaded_model.predict(reshaped_data)
print(predition)
if predition[0]==0:
    print('patient is not diabetic')
else:
     print('patient is diabetic')