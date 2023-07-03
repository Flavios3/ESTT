import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction
import time 

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name        
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) if not test_client else None   
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name
    
    def train(self, metric):
        
        num_train_samples = len(self.dataset)
        self.get_optimizer()           
        self.get_scheduler()

        self.model.train()
        for epoch in range(self.args.num_epochs):
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
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.args.num_epochs, power=self.args.lr_power)
        elif self.args.policy == 'None':
            self.scheduler = None

    def run_epoch(self, metric):

        epoch_loss = 0
        
        start = time.time()
        for cur_step, (images, labels) in enumerate(self.train_loader):   
            self.optimizer.zero_grad()
            images, labels = images.to('cuda:0', dtype=torch.float32), labels.to('cuda:0', dtype=torch.int64)
            outputs = self.model(images)['out']
            loss = self.reduction(self.criterion(outputs, labels), labels)     
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            predicted_labels = torch.argmax(outputs, dim=1)
            metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        end = time.time()

        if self.args.print_times:
            print(f"\n Time for client train epoch: {(end-start):.2f} seconds")

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss /= len(self.train_loader)          
        return epoch_loss 
    
    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())

    def test(self, metric):

        num_samples = len(self.dataset)
        eval_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for cur_step, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to('cuda:0', dtype=torch.float32), labels.to('cuda:0', dtype=torch.int64)
                outputs = self.model(images)['out']                         
                loss = self.reduction(self.criterion(outputs, labels), labels)
                eval_loss += loss.item()
                predicted_labels = torch.argmax(outputs, dim=1)
                metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
            eval_loss /= len(self.test_loader)
        
        return num_samples, eval_loss

class ClientSSL:
    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name           
        self.teacher_model = model
        self.student_model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) if not test_client else None    
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name

    def train(self, metric):

        num_train_samples = len(self.dataset)
        self.get_optimizer()                    
        self.get_scheduler()

        self.student_model.train()
        self.teacher_model.eval()
        for epoch in range(self.args.num_epochs):
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
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=self.args.num_epochs, power=self.args.lr_power)
        elif self.args.policy == 'None':
            self.scheduler = None

    def run_epoch(self, metric):

        epoch_loss = 0

        start = time.time()
        for cur_step, (images, _) in enumerate(self.train_loader):        
            self.optimizer.zero_grad()
            images = images.to('cuda:0', dtype=torch.float32)
            pseudo_labels = self.teacher_model(images)['out']      
            pseudo_labels = torch.argmax(pseudo_labels, dim=1)
            outputs = self.student_model(images)['out']
            loss = self.reduction(self.criterion(outputs, pseudo_labels), pseudo_labels)     
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            predicted_labels = torch.argmax(outputs, dim=1)
            metric.update(pseudo_labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        end = time.time()
        
        if self.args.print_times:
            print(f"\n Time for client train loop: {(end-start):.2f} seconds")

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss /= len(self.train_loader)
        return epoch_loss

    def generate_update(self):
        return copy.deepcopy(self.student_model.state_dict())

    def test(self, metric, metric_pseudo = None):

        num_samples = len(self.dataset)
        eval_loss = 0.0
        eval_loss_pseudo = 0.0

        self.student_model.eval()

        with torch.no_grad():
            for cur_step, (images, labels) in enumerate(self.test_loader):

                images, labels = images.to('cuda:0', dtype=torch.float32), labels.to('cuda:0', dtype=torch.int64)
                outputs = self.student_model(images)['out']
                loss = self.reduction(self.criterion(outputs, labels), labels)
                eval_loss += loss.item()
                predicted_labels = torch.argmax(outputs, dim=1)
                metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())

                if metric_pseudo != None:
                    pseudo_labels = self.teacher_model(images)['out']      
                    pseudo_labels = torch.argmax(pseudo_labels, dim=1)
                    loss_pseudo = self.reduction(self.criterion(outputs, pseudo_labels), pseudo_labels)          
                    eval_loss_pseudo += loss_pseudo.item()
                    metric_pseudo.update(pseudo_labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
               
            eval_loss /= len(self.test_loader)
            eval_loss_pseudo /= len(self.test_loader)
            
        if metric_pseudo != None:
            return num_samples, eval_loss, eval_loss_pseudo 
        
        return num_samples, eval_loss