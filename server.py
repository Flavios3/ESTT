import copy
from collections import OrderedDict,defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from sys import exit

class Server:

    def __init__(self, args, train_clients, test_clients, model, metric):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metric = metric
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.updates = []

    def train(self):

        max_mIoU = 0
        self.warmup_rounds = round(self.args.warmup * self.args.num_rounds)
        self.final_rounds = self.args.num_rounds - self.args.n_final_rounds + 1

        self.train_results, self.val_results, self.test_results_same, self.test_results_diff = defaultdict(list), defaultdict(list),\
            defaultdict(list),defaultdict(list)
        
        for r in range(self.args.num_rounds):

            print(f" Round {r+1}/{self.args.num_rounds}")

            self.metric.reset()
            self.select_clients()

            start = time.time()
            loss_dict = self.train_round()
            end = time.time()
            if self.args.print_times:
                print(f"\n Time for server train round: {(end-start):.2f} seconds")
            train_metrics = self.metric.get_results()

            if (r+1)%self.args.train_display_interval == 0:
                print(" \n Results:")
                print(f"\t - Round_loss={self.round_loss(loss_dict)}")          #round_loss computes the weighted average loss between clients (where the weights are the num_samples for each client)
                print(f"\t - Round_mean_IoU={train_metrics['Mean IoU']}")
                print(f"\t - Round_fw_IoU={train_metrics['FreqW Acc']}")
                print(f"\t - Round_pix_acc={train_metrics['Mean Acc']}")
            self.train_results['loss'].append(self.round_loss(loss_dict))
            self.train_results['mIoU'].append(train_metrics['Mean IoU'])
            self.train_results['fwIoU'].append(train_metrics['FreqW Acc'])
            self.train_results['pAcc'].append(train_metrics['Mean Acc'])

            self.update_model()
            in_final_stage = (r+1)>=self.final_rounds
            if ((r+1)>self.warmup_rounds and (r+1)%self.args.valid_interval==0) or in_final_stage:
                self.eval(test=in_final_stage)
                if self.args.target_miou != 0 and self.val_results['mIoU'][-1] >= self.args.target_miou:
                    print(f"\n Communication rounds: {(r+1) * self.args.clients_per_round}")
                    self.eval_loop(test=True, dom="same")
                    self.eval_loop(test=True, dom="diff")
                    exit()
                if in_final_stage and self.args.save_best_model:
                    if self.val_results['mIoU'][-1] > max_mIoU:
                        max_mIoU = self.val_results['mIoU'][-1] 
                        print("\n New best model found")
                        os.makedirs(os.path.join("Results", "Best_Models", "T2"), exist_ok=True)
                        torch.save(self.model, os.path.join("Results", "Best_Models", "T2", "exp_"+str(self.args.n_exp)+".pth"))

        self.print_results()
        print(" ##### ---- #### Training completed ##### ---- ####")

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        self.selected_clients = np.random.choice(self.train_clients, num_clients, replace=False)

    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)

    def train_round(self):       

        loss_dict = {}

        for i, c in enumerate(self.selected_clients):

            print((f" Client {i + 1}/{len(self.selected_clients)}: {c}"))

            start = time.time()
            self.load_server_model_on_client(c)
            num_samples, update, c_loss = c.train(self.metric)                                                                                    
            self.updates.append((num_samples, update))        
            loss_dict[c.name] = (c_loss, num_samples) 
            end = time.time()

            if self.args.print_times:
                print(f"\n Time for client train round: {(end-start):.2f} seconds")

        return loss_dict
    
    def round_loss(self, loss_dict):
        cum_loss = 0
        tot_samples = 0
        for c_id, (c_loss, c_samples) in loss_dict.items():
            cum_loss += c_loss*c_samples
            tot_samples += c_samples
        return cum_loss/tot_samples
        
    def update_model(self):
        updated_state_dict = self._aggregate()
        self.model.load_state_dict(updated_state_dict, strict=False)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.updates = []

    def _aggregate(self): 

        tot_samples = 0
        accumulate_client = OrderedDict()

        for (c_samples, c_state_dict) in self.updates:
            tot_samples += c_samples
            for k, v in c_state_dict.items():
                if k in accumulate_client:
                    accumulate_client[k] += c_samples*v.type(torch.FloatTensor)
                else:
                    accumulate_client[k] = c_samples*v.type(torch.FloatTensor)

        updated_state_dict = copy.deepcopy(self.model_params_dict)

        for k, v in accumulate_client.items():
            if tot_samples!=0:
                updated_state_dict[k] = v.to('cuda')/tot_samples

        return updated_state_dict
       
    def eval(self, test):
        print(" Evaluation on the training data")
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
        clients = self.train_clients if not test else [self.test_clients[0]] if dom == 'same' else [self.test_clients[1]]
        mode = "Validation" if not test else "Test same dom" if dom=="same" else "Test diff dom"

        loss_dict = dict()
        self.metric.reset()
        
        start = time.time()
        for i, c in enumerate(clients):
            print((f"\n Evaluating client {i+1}/{len(clients)}: {c}"))
            self.load_server_model_on_client(c)
            num_samples, c_loss = c.test(self.metric)              
            loss_dict[c.name] = (c_loss, num_samples) 
        end = time.time()

        if self.args.print_times:
            print(f"\n Time for server {mode} loop: {(end-start):.2f} seconds")

        metric_results = self.metric.get_results()

        print(f"\n {mode} results:")
        print(f"\t - loss={self.round_loss(loss_dict)}")
        print(f"\t - mIoU={metric_results['Mean IoU']}")
        print(f"\t - fwIoU={metric_results['FreqW Acc']}")
        print(f"\t - pAcc={metric_results['Mean Acc']}")
        result_dict['loss'].append(self.round_loss(loss_dict))
        result_dict['mIoU'].append(metric_results['Mean IoU'])
        result_dict['fwIoU'].append(metric_results['FreqW Acc'])
        result_dict['pAcc'].append(metric_results['Mean Acc']) 
    
    def print_results(self):

        #### Plotting Results ####
        
        x_values = [x for x in range(self.warmup_rounds+1, self.final_rounds) if x%self.args.valid_interval==0] + list(range(self.final_rounds, self.args.num_rounds + 1))
        metric_names = ['loss', 'mIoU','fwIoU','pAcc']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            ax.plot(np.arange(1, self.args.num_rounds+1),self.train_results[metric], label='Train')
            ax.plot(x_values, self.val_results[metric], label='Validation')
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            num_ticks = 5 if self.args.num_rounds<=100 else 10 
            ax.set_xticks(list(range(self.args.num_rounds))[::num_ticks])
            ax.set_xticks(list(range(self.args.num_rounds)), minor=True)
            ax.legend()
            ax.grid()
        fig.suptitle(f" T2 Exp. {self.args.n_exp}")
        plt.tight_layout()
        os.makedirs('Results\Plots\Plots_T2', exist_ok=True)
        plt.savefig(os.path.join('Results','Plots','Plots_T2', f'exp_{self.args.n_exp}.png'))
        plt.close(fig)

        #### Computing and printing final results (mu +- std) ####

        print("\n Final mIoU trend:")
        mious = self.train_results['mIoU'][-self.args.n_final_rounds:]
        mean, std = np.mean(mious), np.std(mious)
        print(f"\t Train: (mean, std)=({mean}, {std})")
        mious = self.val_results['mIoU'][-self.args.n_final_rounds:]
        mean, std = np.mean(mious), np.std(mious)
        print(f"\t Validation: (mean, std)=({mean}, {std})")
        if self.args.test:
            mious = self.test_results_same['mIoU'][-self.args.n_final_rounds:]
            mean, std = np.mean(mious), np.std(mious)
            print(f"\t Test same dom: (mean, std)=({mean}, {std})")
            mious = self.test_results_diff['mIoU'][-self.args.n_final_rounds:]
            mean, std = np.mean(mious), np.std(mious)
            print(f"\t Test diff dom: (mean, std)=({mean}, {std})")

        print("\n Final pAcc trend:")
        pacs = self.train_results['pAcc'][-self.args.n_final_rounds:]
        mean, std = np.mean(pacs), np.std(pacs)
        print(f"\t Train: (mean, std)=({mean}, {std})")
        pacs = self.val_results['pAcc'][-self.args.n_final_rounds:]
        mean, std = np.mean(pacs), np.std(pacs)
        print(f"\t Validation: (mean, std)=({mean}, {std})")
        if self.args.test:
            pacs = self.test_results_same['pAcc'][-self.args.n_final_rounds:]
            mean, std = np.mean(pacs), np.std(pacs)
            print(f"\t Test same dom: (mean, std)=({mean}, {std})")
            pacs = self.test_results_diff['pAcc'][-self.args.n_final_rounds:]
            mean, std = np.mean(pacs), np.std(pacs)
            print(f"\t Test diff dom: (mean, std)=({mean}, {std})")


class ServerSSL:

    def __init__(self, args, train_clients, test_clients, teacher_model,student_model,metric):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.teacher_model = teacher_model      
        self.student_model = student_model
        self.metric = metric
        self.metric_pseudo = copy.deepcopy(metric)         
        self.teacher_params_dict = copy.deepcopy(self.teacher_model.state_dict())
        self.student_params_dict = copy.deepcopy(self.student_model.state_dict())
        self.updates = []

    def train(self):

        max_mIoU = 0
        self.warmup_rounds = round(self.args.warmup * self.args.num_rounds)
        self.final_rounds = self.args.num_rounds - self.args.n_final_rounds + 1

        self.train_results, self.val_results, self.val_results_pseudo, self.test_results_same, self.test_results_diff = defaultdict(list), defaultdict(list), defaultdict(list),\
            defaultdict(list),defaultdict(list)
        
        for r in range(self.args.num_rounds):
            
            print(f" Round {r+1}/{self.args.num_rounds}")

            if self.args.teacher_strat == 2 or (self.args.teacher_strat == 3 and (r+1)%self.args.T == 0):
                self.teacher_params_dict = copy.deepcopy(self.student_model.state_dict())
                self.teacher_model.load_state_dict(self.teacher_params_dict)  

            self.metric.reset()
            self.select_clients()

            loss_dict = self.train_round()
            train_metrics = self.metric.get_results()              

            if (r+1)%self.args.train_display_interval == 0:
                print(" \n Results:")
                print(f"\t - Round_loss={self.round_loss(loss_dict)}")        
                print(f"\t - Round_mean_IoU={train_metrics['Mean IoU']}")
                print(f"\t - Round_fw_IoU={train_metrics['FreqW Acc']}")
                print(f"\t - Round_pix_acc={train_metrics['Mean Acc']}")
            self.train_results['loss'].append(self.round_loss(loss_dict))
            self.train_results['mIoU'].append(train_metrics['Mean IoU'])
            self.train_results['fwIoU'].append(train_metrics['FreqW Acc'])
            self.train_results['pAcc'].append(train_metrics['Mean Acc'])

            self.update_model()

            in_final_stage = (r+1)>=self.final_rounds
            if ((r+1)>self.warmup_rounds and (r+1)%self.args.valid_interval==0) or in_final_stage:
                self.eval(test=in_final_stage)
                if in_final_stage and self.args.save_best_model:
                    if self.val_results['mIoU'][-1] > max_mIoU:
                        max_mIoU = self.val_results['mIoU'][-1] 
                        os.makedirs(os.path.join("Results", "Best_Models", "T4"), exist_ok=True)
                        torch.save(self.model, os.path.join("Results", "Best_Models", "T4", "exp_"+str(self.args.n_exp)+".pth"))
            
        self.print_results()
        print(" ##### ---- #### Training completed ##### ---- ####")

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        self.selected_clients = np.random.choice(self.train_clients, num_clients, replace=False)

    def load_server_model_on_client(self, client):
        client.teacher_model.load_state_dict(self.teacher_params_dict)
        client.student_model.load_state_dict(self.student_params_dict)

    def train_round(self):        
        loss_dict = {}
        for i, c in enumerate(self.selected_clients):
            print((f" Client {i + 1}/{len(self.selected_clients)}: {c}"))
            self.load_server_model_on_client(c)
            num_samples, update, c_loss = c.train(self.metric)         
            self.updates.append((num_samples, update))        
            loss_dict[c.name] = (c_loss, num_samples) 
        return loss_dict
    
    def round_loss(self, loss_dict):
        cum_loss = 0
        tot_samples = 0
        for c_id, (c_loss, c_samples) in loss_dict.items():
            cum_loss += c_loss*c_samples
            tot_samples += c_samples
        return cum_loss/tot_samples
        
    def update_model(self):
        updated_state_dict = self._aggregate()
        self.student_model.load_state_dict(updated_state_dict, strict=False)
        self.student_params_dict = copy.deepcopy(self.student_model.state_dict())
        self.updates = []

    def _aggregate(self): 

        tot_samples = 0
        accumulate_client = OrderedDict()

        for (c_samples, c_state_dict) in self.updates:
            tot_samples += c_samples
            for k, v in c_state_dict.items():
                if k in accumulate_client:
                   accumulate_client[k] += c_samples*v.type(torch.FloatTensor)
                else:
                    accumulate_client[k] = c_samples*v.type(torch.FloatTensor)

        updated_state_dict = copy.deepcopy(self.student_params_dict)

        for k, v in accumulate_client.items():
            if tot_samples!=0:
                updated_state_dict[k] = v.to('cuda')/tot_samples
        
        return updated_state_dict
     
    def eval(self, test):
        print(" Evaluation on the training data")
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
        result_dict_pseudo = self.val_results_pseudo
        clients = self.train_clients if not test else [self.test_clients[0]] if dom == 'same' else [self.test_clients[1]]
        mode = "Validation" if not test else "Test same dom" if dom=="same" else "Test diff dom"

        loss_dict, loss_dict_pseudo = dict(), dict()
        self.metric.reset()
        self.metric_pseudo.reset()
        
        for i, c in enumerate(clients):
            print((f"\n Evaluating client {i+1}/{len(clients)}: {c}"))
            self.load_server_model_on_client(c)
            if test == True:
                num_samples, c_loss = c.test(self.metric)  
                loss_dict[c.name] = (c_loss, num_samples)   
            else:
                num_samples, c_loss, c_loss_pseudo = c.test(self.metric,self.metric_pseudo)
                loss_dict[c.name],loss_dict_pseudo[c.name] = (c_loss, num_samples),(c_loss_pseudo, num_samples)
        
        metric_results = self.metric.get_results()
        
        if test == False:
            metric_results_pseudo = self.metric_pseudo.get_results()
        
        print(f"\n {mode} results:")
        print(f"\t - loss={self.round_loss(loss_dict)}")
        print(f"\t - mIoU={metric_results['Mean IoU']}")
        print(f"\t - fwIoU={metric_results['FreqW Acc']}")
        print(f"\t - pAcc={metric_results['Mean Acc']}")
        if test == False:
            print(f"\n {mode} results on pseudo-labels:")
            print(f"\t - loss={self.round_loss(loss_dict_pseudo)}")
            print(f"\t - mIoU={metric_results_pseudo['Mean IoU']}")
            print(f"\t - fwIoU={metric_results_pseudo['FreqW Acc']}")
            print(f"\t - pAcc={metric_results_pseudo['Mean Acc']}")
        
        result_dict['loss'].append(self.round_loss(loss_dict))
        result_dict['mIoU'].append(metric_results['Mean IoU'])
        result_dict['fwIoU'].append(metric_results['FreqW Acc'])
        result_dict['pAcc'].append(metric_results['Mean Acc']) 
        if test == False:
            result_dict_pseudo['loss'].append(self.round_loss(loss_dict_pseudo))
            result_dict_pseudo['mIoU'].append(metric_results_pseudo['Mean IoU'])
            result_dict_pseudo['fwIoU'].append(metric_results_pseudo['FreqW Acc'])
            result_dict_pseudo['pAcc'].append(metric_results_pseudo['Mean Acc'])
            
            
    def print_results(self):

        #### Plotting Results ####
        
        x_values = [x for x in range(self.warmup_rounds+1, self.final_rounds) if x%self.args.valid_interval==0] + list(range(self.final_rounds, self.args.num_rounds + 1))
        metric_names = ['loss', 'mIoU','fwIoU','pAcc']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            ax.plot(np.arange(1, self.args.num_rounds+1),self.train_results[metric], label='Train')
            ax.plot(x_values, self.val_results[metric], label='Val - True')
            ax.set_title(metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            num_ticks = 5 if self.args.num_rounds<=100 else 10 
            ax.set_xticks(list(range(self.args.num_rounds))[::num_ticks])
            ax.set_xticks(list(range(self.args.num_rounds)), minor=True)
            ax.legend()
            ax.grid()
        fig.suptitle(f" T4 Exp. {self.args.n_exp}")
        plt.tight_layout()
        os.makedirs('Results\Plots\Plots_T4', exist_ok=True)
        plt.savefig(os.path.join('Results','Plots','Plots_T4', f'exp_{self.args.n_exp}.png'))
        plt.close(fig)

        #### Computing and printing final results (mu +- std) ####

        print("\n Final mIoU trend:")
        mious = self.train_results['mIoU'][-self.args.n_final_rounds:]
        mean, std = np.mean(mious), np.std(mious)
        print(f"\t Train: (mean, std)=({mean}, {std})")
        mious = self.val_results['mIoU'][-self.args.n_final_rounds:]
        mean, std = np.mean(mious), np.std(mious)
        print(f"\t Validation: (mean, std)=({mean}, {std})")
        if self.args.test:
            mious = self.test_results_same['mIoU'][-self.args.n_final_rounds:]
            mean, std = np.mean(mious), np.std(mious)
            print(f"\t Test same dom: (mean, std)=({mean}, {std})")
            mious = self.test_results_diff['mIoU'][-self.args.n_final_rounds:]
            mean, std = np.mean(mious), np.std(mious)
            print(f"\t Test diff dom: (mean, std)=({mean}, {std})")

        print("\n Final pAcc trend:")
        pacs = self.train_results['pAcc'][-self.args.n_final_rounds:]
        mean, std = np.mean(pacs), np.std(pacs)
        print(f"\t Train: (mean, std)=({mean}, {std})")
        pacs = self.val_results['pAcc'][-self.args.n_final_rounds:]
        mean, std = np.mean(pacs), np.std(pacs)
        print(f"\t Validation: (mean, std)=({mean}, {std})")
        if self.args.test:
            pacs = self.test_results_same['pAcc'][-self.args.n_final_rounds:]
            mean, std = np.mean(pacs), np.std(pacs)
            print(f"\t Test same dom: (mean, std)=({mean}, {std})")
            pacs = self.test_results_diff['pAcc'][-self.args.n_final_rounds:]
            mean, std = np.mean(pacs), np.std(pacs)
            print(f"\t Test diff dom: (mean, std)=({mean}, {std})")