# PyTorch-MNIST

<p>MNIST is a famous dataset of handwritten digits. In this dataset there are 60,000 images of digits for training and 10,000 for test. Also we have used SGD optimizer and cross-entropy function for measuring loss.</p>
<p>We have used simplest neural network. It is a single linear layer and with that alone we got accuracy of 91.87%. Using deep layers can certainly improve the results. </p>

1. Import the required libraries
```Python
import torchvision
import torchvision.transforms as transforms
import torch.utils
import torch.nn as nn
import matplotlib.pyplot as plt
```

2. Load the training and test data from vision datasets and use these in dataloader class for batching.

```Python
train_data = torchvision.datasets.MNIST(root='Data', train= True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='Data', train= False, transform=transforms.ToTensor(), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
```

3. Define your hyperparameters

```Python
batch_size = 100
learning_rate = 0.01
input_size = 28*28
num_classes = 10
num_epochs = 50
```

4. Define your model, loss function and optimizer
```Python
model = nn.Linear(input_size, num_classes)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate)
```

5. Define training loop.
```Python
# Train the model

total_step = len(train_loader)

for epoch in range(num_epochs):
    
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.reshape(-1, input_size)
        
        
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # optimize and backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
```

6. Now evalute on test dataset
```Python
with torch.no_grad():
    total = 0
    correct = 0
    
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
```
