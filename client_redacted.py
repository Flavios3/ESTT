import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils.utils import HardNegativeMining, MeanReduction
import time 

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name            #recall that datasets is a list of IDDADataset, so a single dataset is an instance of IDDADataset 
                                                        #when we define such IDDADatasets, we assign client_name=client_id, where client_id is a key of the 
                                                        #train dict
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) if not test_client else None   
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name
    
    def train(self, metric):
        #take number of training samples
        num_train_samples = len(self.dataset)
        #define optimizer and scheduler
        self.get_optimizer()           
        self.get_scheduler()
        ####lr_step needs to be the one you found in centralized TIMES the number of batches in an epoch of the centralized (i.e., times len(train_loader))
        #set model to train mode
        self.model.train()
        for epoch in range(self.args.num_epochs):
            #call the training loop for the current epoch and return the training loss for such epoch
            client_loss = self.run_epoch(metric)
        update = self.generate_update()
        return num_train_samples, update, client_loss  

    def get_optimizer(self):
        if self.args.opt == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.m)
        elif self.args.opt == 'Adam':
            if self.args.Adam_p == 'dft':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
            elif self.args.Adam_p == 'fd':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.99), eps=10 ** (-1))
            elif self.args.Adam_p == 'mid':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.99), eps=1e-5)
        elif self.args.opt == 'None':
            self.optimizer = None
    
    def get_scheduler(self):
        if self.args.policy == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_step, gamma=self.args.lr_factor)
        elif self.args.policy == 'poly':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                    lr_lambda=lambda cur_iter: (1-cur_iter/self.args.num_epochs) ** self.args.lr_power)
        elif self.args.policy == 'None':
            self.scheduler = None

    def run_epoch(self, metric):
        #init the variable to accumulate the loss
        epoch_loss = 0
        start = time.time()
        #for each batch of the train_loader
        for cur_step, (images, labels) in enumerate(self.train_loader):   
            #zero the grad
            self.optimizer.zero_grad()
            #move data to CUDA
            images, labels = images.to('cuda:0', dtype=torch.float32), labels.to('cuda:0', dtype=torch.int64)
            #compute the output and the loss
            outputs = self.model(images)['out']
            loss = self.reduction(self.criterion(outputs, labels), labels)      #Obtaining the average loss for the considered batch
            #backward pass
            loss.backward()
            #update optmizer and scheduler
            self.optimizer.step()
            #update metric and epoch loss
            epoch_loss += loss.item()
            predicted_labels = torch.argmax(outputs, dim=1)
            metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        end = time.time()
        if self.args.print_times:
            print(f"\n Time for client train epoch: {(end-start):.2f} seconds")
        if self.scheduler is not None:
            self.scheduler.step()
        #compute and print the avg loss
        epoch_loss /= len(self.train_loader)            #Average loss
        return epoch_loss 
    
    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())

    def test(self, metric):
        #take number of training samples
        num_samples = len(self.dataset)
        #init the variable to accumulate the loss
        eval_loss = 0.0
        #set the model to evaluation mode
        self.model.eval()
        #evaluate the performances
        with torch.no_grad():
            for cur_step, (images, labels) in enumerate(self.test_loader):
                #move data to CUDA
                images, labels = images.to('cuda:0', dtype=torch.float32), labels.to('cuda:0', dtype=torch.int64)
                #forward pass and compute loss
                outputs = self.model(images)['out']                         
                loss = self.reduction(self.criterion(outputs, labels), labels)
                #update accumulated loss and metric
                eval_loss += loss.item()
                predicted_labels = torch.argmax(outputs, dim=1)
                metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
            eval_loss /= len(self.test_loader)
        return num_samples, eval_loss


##IF IN SERVER HE EVER UPDATEs THE METRIC, CHANGE THE LOGIC: HERE, WE ARE PASSING THE NEVER-UPDATE ISTANCE OF STREAMSEGMETRIC SAVED IN SERVER.METRIC WHEN
##CALLING CLIENT.TRAIN




class ClientSSL:
    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name            #recall that datasets is a list of IDDADataset, so a single dataset is an instance of IDDADataset 
                                                        #when we define such IDDADatasets, we assign client_name=client_id, where client_id is a key of the 
                                                        #train dict
        self.teacher_model = model
        self.student_model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) if not test_client else None    #SOURCE OF ERRORS
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.ignore_index = 255
        self.conf_thres = 0.9

    def __str__(self):
        return self.name

    def train(self, metric):
        #take number of training samples
        num_train_samples = len(self.dataset)
        #define optimizer and scheduler
        self.get_optimizer()                    
        self.get_scheduler()
        ####lr_step needs to be the one you found in centralized TIMES the number of batches in an epoch of the centralized (i.e., times len(train_loader))
        #set model to train mode
        self.student_model.train()
        self.teacher_model.eval()
        for epoch in range(self.args.num_epochs):
            #call the training loop for the current epoch and return the training loss for such epoch
            client_loss = self.run_epoch(metric)
        update = self.generate_update()
        return num_train_samples, update, client_loss

    def get_optimizer(self):
        if self.args.opt == 'SGD':
            self.optimizer = torch.optim.SGD(self.student_model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, momentum=self.args.m)
        elif self.args.opt == 'Adam':
            if self.args.Adam_p == 'dft':
                self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
            elif self.args.Adam_p == 'fd':
                self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.99), eps=10 ** (-1))
            elif self.args.Adam_p == 'mid':
                self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=self.args.lr, weight_decay=self.args.wd, betas=(0.9, 0.99), eps=1e-5)
        elif self.args.opt == 'None':
            self.optimizer = None
    
    def get_scheduler(self):
        if self.args.policy == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.lr_step, gamma=self.args.lr_factor)
        elif self.args.policy == 'poly':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                    lr_lambda=lambda cur_iter: (1-cur_iter/self.args.num_epochs) ** self.args.lr_power)
        elif self.args.policy == 'None':
            self.scheduler = None

    def run_epoch(self, metric):
        #init the variable to accumulate the loss
        epoch_loss = 0
        start = time.time()
        #for each batch of the train_loader
        for cur_step, (images, _) in enumerate(self.train_loader):         # <-- here it will occurr pickle error
            #zero the grad
            self.optimizer.zero_grad()
            #move data to CUDA
            images = images.to('cuda:0', dtype=torch.float32)
            
            with torch.no_grad():
                out_teacher = self.teacher_model(images)['out']                                                                        
            max_probs, pseudo_labels = torch.max(F.softmax(out_teacher, dim=1), dim=1)  
            pseudo_labels[max_probs < self.conf_thres] = self.ignore_index
            
            #compute the output and the loss
            outputs = self.student_model(images)['out']
            loss = self.reduction(self.criterion(outputs, pseudo_labels), pseudo_labels)      #Compute the loss between the predicted values and the pseudo labels (i.e ground truth)
            #backward pass
            loss.backward()
            #update optmizer and scheduler
            self.optimizer.step()
            #update metric and epoch loss
            epoch_loss += loss.item()
            predicted_labels = torch.argmax(outputs, dim=1)
            metric.update(pseudo_labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        end = time.time()
        
        if self.args.print_times:
            print(f"\n Time for client train loop: {(end-start):.2f} seconds")
        if self.scheduler is not None:
            self.scheduler.step()
        #compute and print the avg loss
        epoch_loss /= len(self.train_loader)
        return epoch_loss

    def generate_update(self):
        return copy.deepcopy(self.student_model.state_dict())

    def test(self, metric, metric_pseudo = None):
        #take number of training samples
        num_samples = len(self.dataset)
        #init the variable to accumulate the loss
        eval_loss = 0.0
        eval_loss_pseudo = 0.0
        #set the model to evaluation mode
        self.student_model.eval()
        #evaluate the performances
        with torch.no_grad():
            for cur_step, (images, labels) in enumerate(self.test_loader):
                #move data to CUDA
                images, labels = images.to('cuda:0', dtype=torch.float32), labels.to('cuda:0', dtype=torch.int64)
                #forward pass and compute loss
                outputs = self.student_model(images)['out']
                loss = self.reduction(self.criterion(outputs, labels), labels)
                #update accumulated loss and metric
                eval_loss += loss.item()
                predicted_labels = torch.argmax(outputs, dim=1)
                metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
                
                if metric_pseudo != None:
                    pseudo_labels = self.teacher_model(images)['out']       #Maybe move to CUDA
                    pseudo_labels = torch.argmax(pseudo_labels, dim=1)
                    loss_pseudo = self.reduction(self.criterion(outputs, pseudo_labels), pseudo_labels)           #Compute the loss between the predicted values and the pseudo labels (i.e ground truth)
                    eval_loss_pseudo += loss_pseudo.item()
                    metric_pseudo.update(pseudo_labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
               
            eval_loss /= len(self.test_loader)
            eval_loss_pseudo /= len(self.test_loader)
            
        if metric_pseudo != None:
            return num_samples, eval_loss, eval_loss_pseudo 
        
        return num_samples, eval_loss