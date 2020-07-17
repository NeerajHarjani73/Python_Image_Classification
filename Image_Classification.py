import numpy as np
from sklearn.datasets import fetch_openml

#mnist_784 is the dataset of images that are 28x28
#Bunch is a dictionary containing objects, key-data container pair
mnist = fetch_openml('mnist_784')

#data table is 2 by 2 array with 70,000 images, each image is represented by a 
#row of 784 numbers, why 784? coz images are 28x28=784 number or 784 features in ML terms
#All 784 numbers are single array, see lab5 YT video
data = mnist['data']
target= mnist['target']

#Graphical tool from matlab
import matplotlib
import matplotlib.pyplot as plt

#A random digit
some_digit = data[36000]
some_digit_image = some_digit.reshape(28,28)

from PIL import Image
im = Image.open('tesla.png')
#tesla_np is 3dimensional array - width,height and depth because image is color image 
tesla_np = np.array(im)

#np.zeroes creates a numpy array full of zeroes
#exmp is 3 dimensional rgba array
exmp = np.zeros([100,200,4], np.uint8)
exmp[: ,:100] = [255, 0, 0, 255]  
exmp[:,100:] = [0, 0, 255, 255]

#Image class is used to convert it into image
myimg = Image.fromarray(exmp) 
myimg.save('myimg3.png')

exmp[:,:100,3]=np.arange(300,400)
exmp[:,100:,3]=np.arange(450,550) 

myimg4 = Image.fromarray(exmp) 
myimg.save('myimg4.png')

tesla_np[:,:,1]=0
tesla2 = Image.fromarray(tesla_np) 
tesla2.save('tesla3.png')

sdi = Image.fromarray(some_digit_image) 
#imshow is image show
plt.imshow(sdi)
plt.axis('off')
plt.show()

target[36000]

#cmap is color map, to put color in variation of number, 5 will be light, 100 will be dark
#range of colors for numbers
#matplotlib.cm.binary is binary colors ( black and white)
#matplotlib.cm.spring is pink and yellow 
#interpolation is blending of colors
plt.imshow(some_digit_image,cmap=matplotlib.cm.spring,interpolation='nearest')
plt.axis('off')
plt.show()

data_train=data[:60000]
target_train = target[:60000]

data_test = data[60000:]
target_test = target[60000:]

np.unique(some_digit ,return_counts=True)

np.random.permutation(4)

shuffle = np.random.permutation(60000)
data_train=data_train[shuffle]
target_train=target_train[shuffle]

target_train_5 = target_train == '5'
np.unique(target_train_5)

from sklearn.linear_model import SGDClassifier 
sgd = SGDClassifier(random_state=100) 
sgd.fit(data_train, target_train_5)
sgd.predict([some_digit])

#This is Cross Validation. we use k-folds here. k folds is number of times
#training data will be spilt, and each spilt will be considered as test
#data to ensure data is not biased
sub_all = np.random.permutation(60000) 

sub_1 = sub_all[0:20000]
sub_2 = sub_all[20000:40000]
sub_3 = sub_all[40000:60000]

sub_1[0:5]
sub_2[0:5]
sub_3[0:5]

d_train_1 = data_train[sub_1]
d_train_2 = data_train[sub_2]
d_train_3 = data_train[sub_3]

t_train_1 = target_train_5[sub_1] 
t_train_2 = target_train_5[sub_2] 
t_train_3 = target_train_5[sub_3]

#axis=0 is used to command that just stack a np array on top of other
#Iteration 1 where second and third subset are training and 1st subset is test 
cd1 = np.append(d_train_2 , d_train_3 , axis=0) 
ct1 = np.append(t_train_2 ,t_train_3 ,axis=0)
sgd.fit(cd1, ct1)
pred1 = sgd.predict(d_train_1)
#Data container must be of same dimension to do vector maths; here pred1 and t_train_1 has the same length
#hence we can do vector maths and use pred1 = t_train_1
#when we apply sum function for true or false, sum functions adds up true as 1 and false as 0
n_correct = sum(pred1==t_train_1) 
n_correct/len(pred1)
#0.96565

#waj = np.arange(5)
#trump = np.array([0,2,2,5,4])
#sum(waj==trump)

#Iteration 2 where first and third subset are training and 2nd subset is test 
cd2 = np.append(d_train_1 , d_train_3 , axis=0) 
ct2 = np.append(t_train_1 ,t_train_3 ,axis=0)
sgd.fit(cd2, ct2)
pred2 = sgd.predict(d_train_2)
n_correct2 = sum(pred2==t_train_2) 
n_correct2/len(pred2)
#0.9613

#Iteration 3 where first and second subset are training and 3rd subset is test 
cd3 = np.append(d_train_1 , d_train_2 , axis=0) 
ct3 = np.append(t_train_1 ,t_train_2 ,axis=0)
sgd.fit(cd3, ct3)
pred3 = sgd.predict(d_train_3)
n_correct3 = sum(pred3==t_train_3) 
n_correct3/len(pred3)
#0.9648

from sklearn.model_selection import cross_val_score
cross_val_score(sgd,data_train ,target_train_5 ,cv=3,scoring='accuracy')

#array([0.9546, 0.9586, 0.9673])

 
pred1[(pred1==True) & (t_train_1==True)] = False
pred2[(pred2==True) & (t_train_2==True)] = False
pred3[(pred3==True) & (t_train_3==True)] = False

n_correct = sum(pred1==t_train_1) 
n_correct/len(pred1)
#0.8952

n_correct = sum(pred2==t_train_2) 
n_correct/len(pred2)
#0.88555

n_correct = sum(pred3==t_train_3) 
n_correct/len(pred3)
#0.9025

from sklearn.model_selection import cross_val_predict
predict_5=cross_val_predict(sgd,data_train,target_train_5,cv=3)
len(predict_5)
np.unique(predict_5)

from sklearn.metrics import confusion_matrix 
confusion_matrix(target_train_5 ,predict_5)
#array([[53433,  1146],
#       [ 1244,  4177]])

#precision score = TP/(TP+FP) -> accuracy of +ve predictions
#recall score = TP/(TP+FN); its also called sensitivity or True positive rate (TPR)
#5s precision = 3939/(3939 + 580)
#non-5s precision = 53999/(53999+1482)
from sklearn.metrics import precision_score , recall_score
precision_score(target_train_5 ,predict_5)
#0.7847078715010333
recall_score(target_train_5 ,predict_5)
#0.7705220439033389

#f1 score is harmonic mean F1 = 2/((1/precision) + (1/recall)) 
#= 2(precision*recall/(precision+recall)) 
#= TP/(TP + ((FN+FP)/2))

from sklearn.metrics import f1_score 
f1_score(target_train_5 ,predict_5)

#0.7775502606105734

#ended at slide 34
class_scores=cross_val_predict(sgd,data_train,target_train_5,cv=3,method='decision_function')
class_scores

from sklearn.metrics import precision_recall_curve 
precisions, recalls, thresholds = precision_recall_curve(target_train_5, class_scores)


#matplotlib example
x=np.array([1,2,3,4,5])
plt.plot(x)
plt.show()

y=x**2
y
plt.plot(x,y,label='y')
plt.legend()
plt.show()

z=np.array([5,4,3,2,1])
plt.plot(x,y,label='y')
plt.plot(x,z,label='z')
plt.legend()
plt.show()
###############

#Precision and recalls have 1 extra value purely for aesthetics; for graphs
len(thresholds)
len(precisions)
len(recalls)

precisions[-1]
recalls[-1]

#x=[1,2,3,4,5]
#x[-3]
#x[:-1]

plt.plot(thresholds,precisions[:-1],label='precisions')
plt.plot(thresholds,recalls[:-1],label='recalls')
plt.legend()
plt.show()

#xlim and ylim are functions to set x or y axis from 0 to 1
plt.plot(recalls,precisions)
plt.xlabel('Recall') 
plt.ylabel('Precision')
plt.ylim([0 ,1])
plt.xlim([0 ,1])
plt.show()

precisions[precisions>=.9]
thresholds[precisions[:-1]>=.9]
threshold_90=thresholds[precisions[:-1]>=.9].min()

predict_5_90=class_scores>threshold_90
np.unique(predict_5_90)

precision_score(target_train_5 ,predict_5_90)
#0.9002812579902838
recall_score(target_train_5 ,predict_5_90)
#0.6495111603025272

#Receiving Operating Characteristics (ROC); it uses recall vs FPR
from sklearn.metrics import roc_curve
fpr, recall, thresholds = roc_curve(target_train_5,class_scores)

#plt.aaxis is similar to lim
plt.plot(fpr,recall)
plt.plot([0,1],[0,1])
plt.axis([0 ,1 ,0 ,1])
plt.xlabel('False Positive Rate') 
plt.ylabel('Recall')
plt.show()

#Diagonal line is the 50% AUC
#AUC is area under curve
from sklearn.metrics import roc_auc_score
roc_auc_score(target_train_5,class_scores)
#.9623642844389065; coz we have few positive 5s
#We use PR curve when we have much less actual 5s; or we care about False positives
#ROC is used otherwise

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=100)

#predict_proba doent give you scores; it gives probability whether it is non 5 or it is 5
forest_scores=cross_val_predict(rfc,data_train,target_train_5,cv=3,method='predict_proba')
forest_scores[0]

fpr_f, recall_f, thresholds_f = roc_curve(target_train_5,forest_scores[:,1])


plt.plot(fpr,recall,label='sgd')
plt.plot([0,1],[0,1])
plt.plot(fpr_f,recall_f,label='rf')
plt.axis([0 ,1 ,0 ,1])
plt.xlabel('False Positive Rate') 
plt.ylabel('Recall')
plt.show()

roc_auc_score(target_train_5,forest_scores[:,1])
#0.9922931870858716

predict_5_rf=cross_val_predict(rfc,data_train,target_train_5,cv=3)
precision_score(target_train_5 ,predict_5_rf)
#0.9871368374362386
recall_score(target_train_5 ,predict_5_rf)
#0.8210662239439218
f1_score(target_train_5 ,predict_5_rf)
#0.8964753272910373

sgd.fit(data_train,target_train)
sgd.predict([some_digit])

#Here we get 4 as the highest value, so some digit is 4
scores = sgd.decision_function([some_digit])
#array([[-836368.4535247 , -461981.66956632, -660256.15197058,
 #       -148855.65250873, -137458.04986937, -154654.76568534,
 #       -864502.26667054, -245167.9063152 , -149510.01775103,
 #       -233700.77221455]])

#argmax gives max values of scores 
np.argmax(scores)

sgd.classes_
#array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

from sklearn.multiclass import OneVsOneClassifier

ovo=OneVsOneClassifier(SGDClassifier(random_state=100))
ovo.fit(data_train,target_train)
ovo.predict([some_digit])
len(ovo.estimators_)
ovo.decision_function([some_digit])
#array([[ 1.5       ,  4.01086892,  0.50210079,  5.22484016,  8.31545536,
#         5.11411311, -0.43998285,  5.13308383,  7.3219439 ,  8.3175768 ]])

cross_val_score(sgd,data_train ,target_train ,cv=3,scoring='accuracy')
#array([0.86552689, 0.86179309, 0.86117918])

import pandas as pd

predict_m=cross_val_predict(sgd,data_train,target_train,cv=3)

ps =precision_score(target_train ,predict_m,average=None)

rs =recall_score(target_train ,predict_m,average=None)

df = pd.DataFrame({'precision':ps,'recall':rs})

precision_score(target_train ,predict_m,average='macro')

recall_score(target_train ,predict_m,average='macro')

precision_score(target_train ,predict_m,average='micro')

recall_score(target_train ,predict_m,average='micro')


rfc.fit(data_train,target_train)
rfc.predict([some_digit])

rfc.predict_proba([some_digit])

predict_rf=cross_val_predict(rfc,data_train,target_train,cv=3)

psrf =precision_score(target_train ,predict_rf,average=None)

rsrf =recall_score(target_train ,predict_rf,average=None)

dfrf = pd.DataFrame({'precision':psrf,'recall':rsrf})

predict_o=cross_val_predict(ovo,data_train,target_train,cv=3)

pso =precision_score(target_train ,predict_o,average=None)

rso =recall_score(target_train ,predict_o,average=None)

dfo = pd.DataFrame({'precision':pso,'recall':rso})

a=np.array([5,8,2,6,10,5])
a.mean()
a.std()


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_train_s = scaler.fit_transform(data_train)

data_train[:5]
#array([[0., 0., 0., ..., 0., 0., 0.],
#       [0., 0., 0., ..., 0., 0., 0.],
#       [0., 0., 0., ..., 0., 0., 0.],
#       [0., 0., 0., ..., 0., 0., 0.],
#       [0., 0., 0., ..., 0., 0., 0.]])


cross_val_score(sgd,data_train_s ,target_train ,cv=3,scoring='accuracy')
#array([0.91016797, 0.90904545, 0.91078662])

predict_ms=cross_val_predict(sgd,data_train_s,target_train,cv=3)

pss =precision_score(target_train ,predict_ms,average=None)

rss =recall_score(target_train ,predict_ms,average=None)

dfs = pd.DataFrame({'precision':pss,'recall':rss})

cnf=confusion_matrix(target_train ,predict_ms)

plt.imshow(cnf, cmap = plt.cm.gray) 
plt.plot()

plt.imshow(cnf, cmap = plt.cm.spring) 
plt.plot()

cnf.sum()
cnf.sum(axis=1)
#array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949])
row_sums=cnf.sum(axis=1,keepdims=True)

cnf_p = cnf/row_sums
plt.imshow(cnf_p, cmap = plt.cm.gray) 
plt.plot()

np.fill_diagonal(cnf_p,0)
plt.imshow(cnf_p, cmap = plt.cm.gray) 
plt.plot()

plt.imshow(cnf_p, cmap = plt.cm.spring) 
plt.plot()