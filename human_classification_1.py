from skimage import feature
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import sklearn
import glob
import numpy as np
import progressbar
from joblib import Parallel, delayed
from sklearn.externals import joblib

print("[Info] Loading Dataset")


train_negative = []
for x in glob.glob("./Dataset/Train/neg/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    img = resize(img, (160, 96), anti_aliasing=True)
    temp = [img,0]
    train_negative.append(temp)    
    # print(img.shape)


train_positive = []
for x in glob.glob("./Dataset/Train/pos/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    img = resize(img, (160, 96), anti_aliasing=True)
    temp = [img,1]
    train_positive.append(temp)
    # print(img.shape)


train_negative_image = np.asarray(train_negative)
print(train_negative_image.shape)
train_positive_image = np.asarray(train_positive)
print(train_positive_image.shape)
train = np.vstack((train_negative_image,train_positive_image))
train = train[1210:1230]
print("[Info] Training Data Shape " + str(train.shape))


test_negative = []
for x in glob.glob("./Dataset/Test/neg/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    img = resize(img, (160, 96), anti_aliasing=True)
    temp = [img,0]
    test_negative.append(temp)    
    # print(img.shape)


test_positive = []
for x in glob.glob("./Dataset/Test/pos/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    img = resize(img, (160, 96), anti_aliasing=True)
    temp = [img,1]
    test_positive.append(temp)
     #print(img.shape)


test_negative_image = np.asarray(test_negative)
print(test_negative_image.shape)
test_positive_image = np.asarray(test_positive)
print(test_positive_image.shape)
test = np.vstack((test_negative_image,test_positive_image))
test = test[445:465]
print("[Info] Test Data Shape " + str(test.shape))
print("[Info] Loading Dataset Complete")

def trainHOG(x):
    (H1, hogImage1)=  feature.hog(x[0], orientations = 3, pixels_per_cell  = (2, 2), cells_per_block  = (2, 2), transform_sqrt=True, block_norm  = 'L1' , visualise=True)
    # print(H1.shape)
    temp = np.append(H1, x[1])
    trainingset.append(temp)

def testHOG(x):
    (H1, hogImage1)=  feature.hog(x[0], orientations = 3, pixels_per_cell  = (2, 2), cells_per_block  = (2, 2), transform_sqrt=True, block_norm  = 'L1' , visualise=True)
    # print(H1.shape)
    temp = np.append(H1, x[1])
    testset.append(temp)

trainingset = []
testset = []


print("[Info] Generating HoG Features Started")
Parallel(n_jobs=6, require='sharedmem')(delayed(trainHOG)(x) for x in progressbar.progressbar(train))
Parallel(n_jobs=6, require='sharedmem')(delayed(testHOG)(x) for x in progressbar.progressbar(test))
print("[Info] Generating HoG Features Completed")


trainingset = np.asarray(trainingset)
# print(trainingset.shape)
testset = np.asarray(testset)
# print(testset.shape)

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

RF = RandomForestClassifier(n_estimators = 50, n_jobs=-1) 
SVM = LinearSVC(random_state=0, tol=1e-5)

# trainingset = np.random.shuffle(trainingset)
np.random.shuffle(trainingset)
# print(trainingset.shape)
TrainX = trainingset[:,0:-1]
TrainY = trainingset[:,-1]
TrainY = np.int_(TrainY)
TrainY_old = np.int_(TrainY)
TrainY = np.eye(2)[TrainY]
TrainY = np.int_(TrainY)


print("[Info] Training Started")
RF.fit(TrainX, TrainY_old)
SVM.fit(TrainX, TrainY_old)
print("[Info] Training completed")

filename = 'RandomForest.sav'
joblib.dump(RF, filename)
filename1 = 'SVM.sav'
joblib.dump(SVM, filename1)

np.random.shuffle(testset)
TestX = testset[:,0:-1]
TestY = testset[:,-1]
TestY = TestY.astype(np.uint8)


RF_loaded = joblib.load(filename)
SVM_loaded = joblib.load(filename1)

print("[Info] Prediction Started")
predicted_labels_RF = RF_loaded.predict(TestX)
predicted_labels_SVM = SVM_loaded.predict(TestX)
print("[Info] Prediction Started")
predicted_labels_RF = predicted_labels_RF.astype(np.uint8)
predicted_labels_SVM = predicted_labels_SVM.astype(np.uint8)


from pandas_ml import ConfusionMatrix
from sklearn.metrics import classification_report, confusion_matrix

TestY = np.int_(TestY)

target_names = ['Negative', 'Positive']
# confusion_matrix = confusion_matrix(TestY, predicted_labels_RF)
confusion_matrix = ConfusionMatrix(TestY, predicted_labels_RF)
# print("Confusion matrix for Random Forest:\n%s" % confusion_matrix)
# print(confusion_matrix.print_stats())
# print("Classification Report for Random Forest")
# classification_report = classification_report(TestY, predicted_labels_RF, target_names=target_names)
# print(classification_report(TestY, predicted_labels_RF, target_names=target_names))
file1 = open("RF.txt","w")
file1.write(str(confusion_matrix)) 
file1.write("\n")
file1.write(str(classification_report(TestY, predicted_labels_RF, target_names=target_names))) 
file1.write("\n")
file1.write(str(confusion_matrix.stats())) 
file1.close() 
print("[Info] Results Written for RF on File")

target_names = ['Negative', 'Positive']
confusion_matrix = ConfusionMatrix(TestY, predicted_labels_SVM)
# print("Confusion matrix for SVM:\n%s" % confusion_matrix)
# print(confusion_matrix.print_stats())
# print("Classification Report for SVM")
# classification_report = classification_report(TestY, predicted_labels_SVM, target_names=target_names)
# print(classification_report(TestY, predicted_labels_SVM, target_names=target_names))
file2 = open("SVM.txt","w")
file2.write(str(confusion_matrix)) 
file2.write("\n")
file2.write(str(classification_report(TestY, predicted_labels_SVM, target_names=target_names)))  
file2.write("\n")
file2.write(str(confusion_matrix.stats())) 
file2.close() 
print("[Info] Results Written for SVM on File")



