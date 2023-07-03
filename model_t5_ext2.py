from main import *
from utils.args import *
from models.deeplabv3 import *
from models.mobilenetv2 import *
from datasets.idda import *
from FDA.fda import * 

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import matplotlib.pyplot as plt
import time

from torch.utils.data import ConcatDataset,DataLoader

####################################################################################################################################################

class Trainer:

    def __init__(self, args):
        self.args = args
        self.metric = set_metrics(self.args)
        self.model = model_init(args).cuda()
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())

    def train(self):
        max_mIoU = 0
        self.warmup_epochs = round(self.args.warmup * self.args.num_epochs)
        self.final_epochs = self.args.num_epochs - self.args.n_final_epochs + 1
        
        self.generate_datasets()
        
        print(" Creating the styles")
        self.style_augment = StyleAugmentExtended(self.args)
        for client_ds in self.client_datasets:
            self.style_augment.add_style(client_ds, name = client_ds.client_name)
        print("\n Finished!")
        if self.args.mode == 'interpolation':
            self.style_augment.create_bank_clustered()
        self.train_loader.dataset.style_augment = self.style_augment

        self.train_results, self.val_results = defaultdict(list), defaultdict(list)
        self.test_results_same, self.test_results_diff = defaultdict(list), defaultdict(list)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255,reduction = 'none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.get_optimizer()
        self.get_scheduler()

        for epoch in range(self.args.num_epochs):
            print(f"\n\n Epoch {epoch+1}/{self.args.num_epochs}")

            self.train_epoch()
            if (epoch+1)%self.args.train_display_interval == 0:
                self.print_train_stats()

            in_final_stage = (epoch+1)>=self.final_epochs
            if ((epoch+1)>self.warmup_epochs and (epoch+1)%self.args.valid_interval==0) or in_final_stage:
                self.eval(test=in_final_stage)
                if in_final_stage and self.args.save_best_model:
                    if self.val_results['mIoU'][-1] > max_mIoU:
                        max_mIoU = self.val_results['mIoU'][-1] 
                        print("\n New best model found")
                        os.makedirs(os.path.join("Results", "Best_Models", "T3"), exist_ok=True)
                        torch.save(self.model, os.path.join("Results", "Best_Models", "T3", "exp_"+str(self.args.n_exp)+".pth"))

        self.print_results()
            
    def generate_datasets(self):

        print(' Generating datasets...')
        train_dataset, valid_datasets, test_datasets = get_datasets(self.args)
        self.client_datasets = valid_datasets if self.args.style_transfer else None
        valid_dataset = ConcatDataset(valid_datasets)
        print(' ...done')

        self.valid_loader = DataLoader(valid_dataset, batch_size=self.args.bs, shuffle=False, num_workers=self.args.nw)
        if self.args.test:
            self.test_loader_samedom = DataLoader(test_datasets[0], batch_size=self.args.bs, shuffle=False, num_workers=self.args.nw)
            self.test_loader_diffdom = DataLoader(test_datasets[1], batch_size=self.args.bs, shuffle=False, num_workers=self.args.nw)
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.bs, shuffle=True, num_workers=self.args.nw, drop_last=True)


    def reload_initial_setup(self):
        print(" Reloading initial param.s...")
        set_seed(self.args.seed)
        self.model = model_init(self.args).cuda()
        print(" ...done")
        

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
        elif self.args.policy == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.num_epochs, eta_min=0, last_epoch=-1, verbose=False)
        elif self.args.policy == 'None':
            self.scheduler = None

    def train_epoch(self):

        train_loss = 0.0
        self.metric.reset()
        self.model.train()

        start_time = time.time()
        for i, (images, labels) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images, labels = images.to("cuda:0", dtype=torch.float32), labels.to("cuda:0", dtype=torch.int64)
            outputs = self.model(images)["out"]  
            predicted_labels = torch.argmax(outputs, dim=1)  
            loss = self.reduction(self.criterion(outputs, labels),labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            self.metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())
        end_time = time.time()

        if self.args.print_times:
            print(f"\n Time for train loop: {(end_time-start_time):.2f} seconds")

        if self.scheduler is not None:
            self.scheduler.step()

        metric_results = self.metric.get_results()
        train_loss /= len(self.train_loader)
        self.train_results['loss'].append(train_loss)
        self.train_results['mIoU'].append(metric_results['Mean IoU'])
        self.train_results['fwIoU'].append(metric_results['FreqW Acc'])
        self.train_results['pAcc'].append(metric_results['Mean Acc'])

    def print_train_stats(self):
        print("\n Train results for current epoch: ")
        print("\t loss: ", self.train_results['loss'][-1])
        print("\t mIoU: ", self.train_results['mIoU'][-1])
        print("\t fwIoU: ", self.train_results['fwIoU'][-1])
        print("\t pAcc: ", self.train_results['pAcc'][-1])

    def eval(self,test):
        
        start_time = time.time()
        self.eval_loop(test=False)
        end_time = time.time()
        if self.args.print_times:
            print(f"\n Time for eval loop: {(end_time-start_time):.2f} seconds")
        if test and self.args.test:
            self.eval_loop(test=True, dom="same")
            self.eval_loop(test=True, dom="diff")
        
    def eval_loop(self, test, dom="same"):

        result_dict = self.val_results if not test else self.test_results_same if dom == "same" else self.test_results_diff
        loader = self.valid_loader if not test else self.test_loader_samedom if dom == 'same' else self.test_loader_diffdom
        mode = "Validation" if not test else "Test same dom" if dom=="same" else "Test diff dom"

        eval_loss = 0.0
        self.metric.reset()
        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to("cuda:0", dtype=torch.float32), labels.to("cuda:0", dtype=torch.int64)
                outputs = self.model(images)["out"]                            
                predicted_labels = torch.argmax(outputs, dim=1)           
                loss = self.reduction(self.criterion(outputs, labels),labels)
                eval_loss += loss.item()
                self.metric.update(labels.detach().cpu().numpy(), predicted_labels.detach().cpu().numpy())  
        metric_results = self.metric.get_results()
        eval_loss /= len(loader)

        print(f"\n {mode} results:")
        print(f"\t - loss={eval_loss}")
        print(f"\t - mIoU={metric_results['Mean IoU']}")
        print(f"\t - fwIoU={metric_results['FreqW Acc']}")
        print(f"\t - pAcc={metric_results['Mean Acc']}")
        result_dict['loss'].append(eval_loss)
        result_dict['mIoU'].append(metric_results['Mean IoU'])
        result_dict['fwIoU'].append(metric_results['FreqW Acc'])
        result_dict['pAcc'].append(metric_results['Mean Acc'])    
    
    def print_results(self):

        #### Plotting Results ####

        x_values = [x for x in range(self.warmup_epochs+1, self.final_epochs) if x%self.args.valid_interval==0] + list(range(self.final_epochs, self.args.num_epochs + 1))
        metric_names = ['loss', 'mIoU','fwIoU','pAcc']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            ax.plot(np.arange(1, self.args.num_epochs+1),self.train_results[metric], label='Train')
            ax.plot(x_values, self.val_results[metric], label='Validation')
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            num_ticks = self.args.num_epochs//10 
            ax.set_xticks(list(range(self.args.num_epochs)), minor=True)
            ax.set_xticks(list(range(self.args.num_epochs))[::num_ticks])
            ax.legend()
            ax.grid()
        fig.suptitle(f" T3 Exp. {self.args.n_exp}")
        plt.tight_layout()
        os.makedirs(os.path.join('Results','Plots','Plots_T5.2'), exist_ok=True)
        plt.savefig(os.path.join('Results','Plots','Plots_T5.2', f'exp_{self.args.n_exp}.png'))
        plt.close(fig)

        #### Computing and printing final results (mu +- std) ####

        print("\n Final mIoU trend:")
        mious = self.train_results['mIoU'][-self.args.n_final_epochs:]
        mean, std = np.mean(mious), np.std(mious)
        print(f"\t Train: (mean, std)=({mean}, {std})")
        mious = self.val_results['mIoU'][-self.args.n_final_epochs:]
        mean, std = np.mean(mious), np.std(mious)
        print(f"\t Validation: (mean, std)=({mean}, {std})")
        if self.args.test:
            mious = self.test_results_same['mIoU'][-self.args.n_final_epochs:]
            mean, std = np.mean(mious), np.std(mious)
            print(f"\t Test same dom: (mean, std)=({mean}, {std})")
            mious = self.test_results_diff['mIoU'][-self.args.n_final_epochs:]
            mean, std = np.mean(mious), np.std(mious)
            print(f"\t Test diff dom: (mean, std)=({mean}, {std})")

        print("\n Final pAcc trend:")
        pacs = self.train_results['pAcc'][-self.args.n_final_epochs:]
        mean, std = np.mean(pacs), np.std(pacs)
        print(f"\t Train: (mean, std)=({mean}, {std})")
        pacs = self.val_results['pAcc'][-self.args.n_final_epochs:]
        mean, std = np.mean(pacs), np.std(pacs)
        print(f"\t Validation: (mean, std)=({mean}, {std})")
        if self.args.test:
            pacs = self.test_results_same['pAcc'][-self.args.n_final_epochs:]
            mean, std = np.mean(pacs), np.std(pacs)
            print(f"\t Test same dom: (mean, std)=({mean}, {std})")
            pacs = self.test_results_diff['pAcc'][-self.args.n_final_epochs:]
            mean, std = np.mean(pacs), np.std(pacs)
            print(f"\t Test diff dom: (mean, std)=({mean}, {std})")
            



####################################################################################################################################################

def get_parser(args_dict):
    #define the parser
    parser = argparse.ArgumentParser()
    #arguments for the experimental setup
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'gtaV'], help='dataset name')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--n_exp', type=int, help='number of the experiment')
    parser.add_argument('--print_times', type=bool, default=False, help='whether to print the time needed for a train/eval loop')
    parser.add_argument('--mode', type=str, choices=['interpolation', 'noise'], help='how to handle the styole applciation')
    parser.add_argument('--p_interpolation', type=float, default=0.5, help='probability of applying the style interpolation')
    #arguments for the model parameters  
    parser.add_argument('--da', type=str, choices=['basic', 'advanced'], help='type of data augmentation')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--nw', type=int, default=2, help='num workers')
    parser.add_argument('--lr', type=float, help='lr')
    parser.add_argument('--wd', type=float, help='wd')
    #arguments for the criterion, optimizer and scheduler
    parser.add_argument('--hnm', type=bool, help='whether to use hardnegativemining or mean reduction')
    parser.add_argument('--opt', type=str, choices=['None', 'SGD', 'Adam'], help='optimizer')
    parser.add_argument('--m', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--policy', type=str, choices=['None', 'step', 'poly', 'cosine'], help='scheduler')
    parser.add_argument('--lr_power', type=float, default=0.9, help='poly scheduler power')
    parser.add_argument('--lr_step', type=int, default=15, help='step scheduler decay step')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='step scheduler decay factor')
    parser.add_argument('--Adam_p', type=str, choices=['dft', 'fd', 'mid'], help='set Adams parameter to the default or FedDrive ones, or try a middle ground')
    #arguments for the handling of validation and testing
    parser.add_argument('--test', type=bool, default=True, help='whether to test')
    parser.add_argument('--warmup', type=float, default=0.5, help='percentage of the total epochs/rounds to be considered as warmup (only training, no validation)')
    parser.add_argument('--train_display_interval', type=int, default=5, help='epoch/round interval for the display of the train statistics')
    parser.add_argument('--valid_interval', type=int, default=5, help='epoch/round interval for a validation loop')
    parser.add_argument('--n_final_epochs', type=int, default=5, help='number of final epochs, on which to densily evaluate')
    parser.add_argument('--save_best_model', type=bool, default=True, help='whether to save the best model among the last final_epochs')
    #arguments for FDA
    parser.add_argument('--style_transfer',type=bool,default=False,help='flag used to perform FDA')
    parser.add_argument('--L',type=float,default=0.1,help='beta parameter of FDA')
    parser.add_argument('--num_images_per_style',type=int,default=-1,help='number of images to compute the style, -1 to use all images')
    parser.add_argument('--n_cl', type=int, default=15, help='maximum value (of k) for the number of clusters')
    parser.add_argument('--m_cl', type=int, default=2, help='minimum value (of k) for the number of clusters')
    parser.add_argument('--N_cl', type=int, default=15, help=' number of iterations of KMeans for a fixed k')

    args = parser.parse_args(args=[])
    for k, v in args_dict.items():
        setattr(args, k, v)
        
    return args

def print_setup(args):
    print(" \n\n ############################################# \n\n I'm in the experiment:\n ",args)

####################################################################################################################################################

def experiment_t5_ext2(args_dict):
    #create the parser
    args = get_parser(args_dict)
    print_setup(args)
    #set the seed
    set_seed(args.seed)
    #start the experiment
    trainer = Trainer(args)
    trainer.train()