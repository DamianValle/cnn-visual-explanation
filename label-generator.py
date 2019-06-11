import os
import json
import numpy as np

X_train = []
y_train = []
label_count = 0
nolabel_count = 0
i = 0
for i in range(1, 58000):
    i = i+1
    if(i%100==0):
        print(i)
    if(i%1000==0):
        print('label count:{}\tNo label count:{}'.format(label_count, nolabel_count))
    path = '<rico root path>/semantic_annotations/{}.json'.format(i)
    if(os.path.isfile(path)):
        with open(path) as json_file:
            legend = json.load(json_file)
            if("'iconClass': '<target label>'" in str(legend['children'])):
                if(label_count < 5000):
                    X_train.append(i)
                    label_count += 1
                    y_train.append(1)
            elif(nolabel_count < 5000):
                    X_train.append(i)
                    nolabel_count += 1
                    y_train.append(0)

print('label count:{}\tNo label count:{}'.format(label_count, nolabel_count))
print(np.shape(X_train))
print(np.shape(y_train))

#Shuffle dataset
X_train = np.array(X_train)
y_train = np.array(y_train)

s = np.arange(y_train.shape[0])
np.random.shuffle(s)

X_train = X_train[s]
y_train = y_train[s]

with open('train/labels/X_train_<target label>.txt', 'w+') as outfile:  
    json.dump(X_train.tolist(), outfile)

with open('train-labels/y_train_<target label>.txt', 'w+') as outfile:  
    json.dump(y_train.tolist(), outfile)















