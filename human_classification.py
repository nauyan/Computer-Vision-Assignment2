from skimage import feature
from skimage import io, color
import sklearn
import glob
import numpy as np
import progressbar


print("[Info] Loading Dataset")
train_negative = []
for x in glob.glob("./Dataset/Train/neg/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    temp = [img,0]
    train_negative.append(temp)    
    # print(img.shape)


train_positive = []
for x in glob.glob("./Dataset/Train/pos/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    temp = [img,1]
    train_positive.append(temp)
    # print(img.shape)


train_negative_image = np.asarray(train_negative)
# print(train_negative_image.shape)
train_positive_image = np.asarray(train_positive)
# print(train_positive_image.shape)
train = np.vstack((train_negative_image,train_positive_image))
print("Training Data Shape " + str(train.shape))


test_negative = []
for x in glob.glob("./Dataset/Test/neg/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    temp = [img,0]
    test_negative.append(temp)    
    # print(img.shape)


test_positive = []
for x in glob.glob("./Dataset/Test/pos/*"):
    img = io.imread(x)
    img = color.rgb2gray(img)
    temp = [img,1]
    test_positive.append(temp)
    # print(img.shape)


test_negative_image = np.asarray(test_negative)
# print(test_negative_image.shape)
test_positive_image = np.asarray(test_positive)
# print(test_positive_image.shape)
test = np.vstack((test_negative_image,test_positive_image))
print("Test Data Shape " + str(test.shape))


trainingset = []
for x in progressbar.progressbar(train):
    (H1, hogImage1)=  feature.hog(x[0], orientations = 3, pixels_per_cell  = (2, 2), cells_per_block  = (2, 2), transform_sqrt=True, block_norm  = 'L1' , visualise=True)
    #print(H1.shape)
    temp = [H1,x[1]]
    trainingset.append(temp)
trainingset = np.asarray(trainingset)
print(trainingset.shape)


testset = []
for x in progressbar.progressbar(test):
    (H1, hogImage1)=  feature.hog(x[0], orientations = 3, pixels_per_cell  = (2, 2), cells_per_block  = (2, 2), transform_sqrt=True, block_norm  = 'L1' , visualise=True)
    #print(H1.shape)
    temp = [H1,x[1]]
    testset.append(temp)
testset = np.asarray(testset)
print(testset.shape)
"""
print("[Info] Loading Complete")
print("Number of Training Samples " + str(len(train_negative)+len(train_positive)))
print("Number of Test Samples " + str(len(test_negative)+len(test_positive)))

train_negative_image = np.asarray(train_negative)
train_negative_label = np.zeros(len(train_negative))
#print(train_negative_image.shape)
#print(train_negative_label.shape)
#train_negative_array = np.vstack((train_negative_image,train_negative_label))
# print(train_negative_array.shape)
#train_negative_array = train_negative_array.reshape(len(train_negative),2)

train_positive_image = np.asarray(train_positive)
train_positive_label = np.ones(len(train_positive))
#print(train_positive_image.shape)
#print(train_positive_label.shape)
# train_positive_array = np.vstack((train_positive_image,train_positive_label))
# train_positive_array = train_negative_array.reshape(len(train_positive),2)

#train = np.hstack((train_negative_array,train_positive_array))
# print(train_negative_array.shape)
# print(train_positive_array.shape)
# print(train_negative_label.shape[0])
# train_images = np.hstack((train_negative_image,train_positive_image))
# train_images = np.concatenate((train_negative_image,train_positive_image[:,None]),axis=0)
# train_labels = np.hstack((train_negative_label,train_positive_label))
# print(train_images.shape)
# print(train_labels.shape)

print("[Info] Generating HoG Features for Negative Training Set")
trainX_negative = []
#for i in progressbar.progressbar((range(train_negative_label.shape[0])):
for i in progressbar.progressbar(range(10)):
    (H1, hogImage1)=  feature.hog(train_negative_image[i], orientations = 3, pixels_per_cell  = (2, 2), cells_per_block  = (2, 2), transform_sqrt=True, block_norm  = 'L1' , visualise=True)
    # print(H1.shape)
    trainX_negative.append(H1)
print("[Info] Generating HoG Features for Negative Training Set Complete")

print("[Info] Generating HoG Features for Positive Training Set")
trainX_positive = []
#for i in progressbar.progressbar((range(train_positive_label.shape[0])):
for i in progressbar.progressbar(range(10)):
    (H1, hogImage1)=  feature.hog(train_positive_image[i], orientations = 3, pixels_per_cell  = (2, 2), cells_per_block  = (2, 2), transform_sqrt=True, block_norm  = 'L1' , visualise=True)
    print(H1.shape)
    trainX_positive.append(H1)
print("[Info] Generating HoG Features for Positive Training Set Complete")

#trainX_negative = np.asarray(trainX_negative)
#print(trainX_negative.shape)
trainX_positive = np.asarray(trainX_positive)
print(trainX_positive.shape)
"""
