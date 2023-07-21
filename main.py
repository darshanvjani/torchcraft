import torch
import torchvision
import torchvision.transforms as transforms
import albumentations
import numpy as np
# from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets , transforms 
import torchvision

from torchcraft.dataloader import albumentation as A 

from torchcraft.utils.helper import *
from torchcraft.utils.gradcam import *
from torchcraft.utils.plot_metrics import *
from torchcraft.utils.test import *
from torchcraft.utils.train import *

from torchcraft.models import resnet


class main():

    def __init__(self,device):
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.device = device
        self.train_losses = []
        self.test_losses = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.plot_train_acc=[]
        self.lrs=[]
        pass

    def dataloading_aubumentation(self,mean,std,batch_size):
        albu_obj = A.CIFAR10Albumentation()
        train_transform = albu_obj.train_transform(mean,std)
        test_transform = albu_obj.test_transform(mean,std)

        trainset = torchvision.datasets.CIFAR10(root='/content',train=True,download=True,transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='/content',train=False,download=True,transform=test_transform)
                
        train_dataloader = torch.utils.data.DataLoader(trainset,num_workers=2,shuffle=True,batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(testset,num_workers=2,shuffle=True,batch_size=batch_size)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.trainset = trainset
        self.testset = testset
    
    def show_augmented_img(self,no_of_images):
        helper.plot_images(self.trainset,no_of_images,self.classes)

    def model(self,model_name,set_seed_no,show_summery):
        if model_name == 'resnet34':
            net = resnet.ResNet34()
            self.net = net
        if set_seed_no != None:
            set_seed(set_seed_no,True)
        if show_summery == True:
            model_summary(self.net,(3,32,32))
        return net

    def train_model(self,optimizer,epochs,lam_reg,schedular,criterian,show_plots=True):
        for epoch in range(epochs):
            train(self.net,self.device,self.train_dataloader,optimizer,epoch,self.train_accuracy,self.train_losses,lam_reg,schedular,criterian,self.lrs)
            test(self.net,self.device,self.test_dataloader,self.test_accuracy,self.test_losses,criterian)
        if show_plots==True:
            plot_metrics([self.train_accuracy,self.train_losses,self.test_accuracy,self.test_losses])
            conf_matrix = compute_confusion_matrix(self.net,self.test_dataloader,self.device)
            plot_confusion_matrix(conf_matrix)
        
    def examination(self,no_of_images):
        wrong_pred = wrong_predictions(self.net,self.test_dataloader,no_of_images,self.device,self.classes)
        target_layers = ["layer1","layer2","layer3","layer4"]
        gradcam_output, probs, predicted_classes = generate_gradcam(wrong_pred[:10],self.net,target_layers,self.device)
        plot_gradcam(gradcam_output, target_layers, self.classes, (3, 32, 32),predicted_classes, wrong_pred[:10])

        






