# Develop a Convolutional Deep Neural Network for Image Classification
## NAME : KARTHIKRAJ C
## REG.NO : 212224230117

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### Step 1: 
Load the Fashion-MNIST training and testing datasets and preprocess the images by converting them into tensors and normalizing the pixel values.

### Step 2: 
Create DataLoaders to divide the datasets into batches for efficient training and testing.

### Step 3:
Build a CNN model with three convolutional layers, ReLU activation, max-pooling layers, and fully connected layers for classifying images into 10 categories.

### Step 4: 
Initialize the Cross-Entropy Loss function and Adam optimizer, then train the CNN for multiple epochs using forward propagation, loss calculation, backpropagation, and weight updates.

### Step 5: 
Evaluate the trained model using the test dataset and calculate the classification accuracy.

### Step 6: 
Generate a confusion matrix and classification report to analyze the model's prediction performance using precision, recall, and F1-score.

### Step 7: 
Select a single test image, pass it through the trained CNN, and display the actual class and predicted class.





## PROGRAM
```
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])

train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=transform)
test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,transform=transform)

im,lbl=train_set[0]
print(im.shape)
print(len(train_set))
print(len(test_set))


from torch.utils.data import DataLoader
trl=DataLoader(train_set,batch_size=32,shuffle=True)
tstl=DataLoader(test_set,batch_size=32,shuffle=False)

class CNNclassifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.c2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.c3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.l1=nn.Linear(128*3*3,64)
        self.l2=nn.Linear(64,32)
        self.l3=nn.Linear(32,10)
    def forward(self,x):
        x=self.pool(torch.relu(self.c1(x)))
        x=self.pool(torch.relu(self.c2(x)))
        x=self.pool(torch.relu(self.c3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.l1(x))
        x=torch.relu(self.l2(x))
        x=self.l3(x)
        return x



model=CNNclassifier1()
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    device=torch.device('cuda')
    model.to(device)


summary(model,input_size=(1,28,28))



epochs=3
running_loss=0.0

for i in range(epochs):
    model.train()
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b)
        loss.backward()
        op.step()
        running_loss+=loss.item()
    print(f"Loss:{i}",running_loss/len(trl))


t=0
c=0
act=[]
pre=[]
model.eval()
with torch.no_grad():
    for img,labels in tstl:
        output=model(img)
        _,predicted=torch.max(output,1)
        t=t+labels.size(0)
        c+=(predicted==labels).sum().item()
        pre.extend(predicted.cpu().numpy())
        act.extend(labels.cpu().numpy())
accuracy=c/t*100
print("Accuracy Score:",accuracy)
conf_matrix=confusion_matrix(act,pre)
class_report=classification_report(act,pre,target_names=test_set.classes)
print("Classification Report:",class_report)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues',xticklabels=test_set.classes,yticklabels=test_set.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


with torch.no_grad():
    img1,label=test_set[0]
    output=model(img1.unsqueeze(0))
    _,pred=torch.max(output,1)
    classes=test_set.classes
    img1 = img1 * 0.5 + 0.5
    plt.imshow(img1.squeeze(),cmap="gray")
    plt.title("Predicted Image")
    plt.axis('off')
    plt.show()
    print(f"Actual:{classes[label]}\n")
    print(f"Predicted:{classes[pred.item()]}\n")



```


### OUTPUT

## Training Loss per Epoch

<img width="547" height="122" alt="image" src="https://github.com/user-attachments/assets/4036bade-d452-43ab-a2b3-3d1745ab9b7b" />


## Confusion Matrix

<img width="528" height="448" alt="image" src="https://github.com/user-attachments/assets/a6b477c5-e3df-4c67-b4c3-70ddf1a94ba7" />


## Classification Report
<img width="703" height="322" alt="image" src="https://github.com/user-attachments/assets/53dbd207-b681-4cc4-b3cf-8a8c263d0300" />


### New Sample Data Prediction
<img width="591" height="502" alt="image" src="https://github.com/user-attachments/assets/9686510a-c930-4830-86b4-833427f5a12e" />

## RESULT

Thus the Development of  a Convolutional Deep Neural Network for Image Classification is successfull.
