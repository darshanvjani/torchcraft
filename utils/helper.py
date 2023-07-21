import torch
import matplotlib.pyplot as plt
from torchsummary import summary
import yaml
from pprint import pprint
import random
import numpy as np
import torch.nn as nn
from torchvision import datasets, transforms
from itertools import product



def imshow(img):
    # functions to show an image
    fig, ax = plt.subplots(figsize=(12, 12))
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_images(trainset,N_IMAGES,classes,normalize = True):

    images, labels = zip(*[(image, label) for image, label in 
                           [trainset[i] for i in range(N_IMAGES)]])

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes[labels[i]]
        ax.set_title(label)
        ax.axis('off')

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image    
    
def unnormalize(img):
  mean = (0.49139968, 0.48215841, 0.44653091)
  std = (0.24703223, 0.24348513, 0.26158784)
#   mean,std = calculate_mean_std("CIFAR")
  img = img.cpu().numpy().astype(dtype=np.float32)
  
  for i in range(img.shape[0]):
    img[i] = (img[i]*std[i])+mean[i]
  
  return np.transpose(img, (1,2,0))
  
  
def calculate_mean_std(dataset):
    if dataset == 'CIFAR10':
        train_transform = transforms.Compose([transforms.ToTensor()])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0,1,2))/255
        std = train_set.data.std(axis=(0,1,2))/255
        return mean, std

def get_mean_std(loader):
  #VAR[x] = E[x**2] - E[x]**2

  channels_sum, channels_squared_sum, num_batches = 0, 0, 0

  for data, _ in loader:
    channels_sum += torch.mean(data, dim = [0,2,3])
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
    num_batches += 1

  mean = channels_sum/num_batches
  std = (channels_squared_sum / num_batches - mean**2)**0.5
  
  return mean , std

def set_seed(seed,cuda_available):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_available:
        torch.cuda.manual_seed(seed)
    
    
def process_config(file_name):
    with open(file_name, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            print(" loading Configuration of your experiment ..")
            return config
        except ValueError:
            print("INVALID yaml file format.. Please provide a good yaml file")
            exit(-1)


def model_summary(model, input_size):
    result = summary(model.cuda(), input_size=input_size)
    print(result)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def class_level_accuracy(model, loader, device, classes):

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def compute_confusion_matrix(model, data_loader, device):

    all_targets, all_predictions = [], []
    with torch.no_grad():

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets
            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to('cpu'))
            all_predictions.extend(predicted_labels.to('cpu'))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
        
    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))
    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([0, class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])
    n_labels = class_labels.shape[0]
    lst = []
    z = list(zip(all_targets, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))
    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat

def wrong_predictions(model,test_loader,num_img,device,classes):
    wrong_images=[]
    wrong_label=[]
    correct_label=[]
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)        
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])  
      
            wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
        print(f'Total wrong predictions are {len(wrong_predictions)}')
            
        plot_misclassified(wrong_predictions,num_img,test_loader,classes)
      
    return wrong_predictions

def plot_misclassified(wrong_predictions,num_img,dataloader,classes):
    fig = plt.figure(figsize=(15,12))
    fig.tight_layout()
    mean,std = get_mean_std(dataloader)
    mean,std = torch.tensor(mean),torch.tensor(std)
    for i, (img, pred, correct) in enumerate(wrong_predictions[:num_img]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        img = torch.tensor(img)
        for j in range(img.shape[0]):
            img[j] = (img[j]*std[j])+mean[j]
            
        img = np.transpose(img, (1, 2, 0)) 
        ax = fig.add_subplot(5, 5, i+1)
        fig.subplots_adjust(hspace=.5)
        ax.axis('off')
        # self.class_names,_ = self.get_classes()
            
        ax.set_title(f'\nActual : {classes[target.item()]}\nPredicted : {classes[pred.item()]}',fontsize=10)  
        ax.imshow(img)  
          
    plt.show()