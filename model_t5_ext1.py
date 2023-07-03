from main import *
import argparse
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

        self.warmup_epochs = round(self.args.warmup * self.args.num_epochs)
        self.final_epochs = self.args.num_epochs - self.args.n_final_epochs + 1

        self.generate_datasets()        

        print(" Creating the styles")
        self.style_augment = StyleAugmentClustered(self.args)
        for client_ds in self.client_datasets:
            self.style_augment.add_style(client_ds, name = client_ds.client_name)
        print("\n Finished!")
        self.style_augment.create_clustering()        
        self.train_loader.dataset.style_augment = self.style_augment
        
        self.states_dict = dict()
        self.train_results = dict()
        for i in range(len(self.style_augment.cluster_mapping)):
            self.train_results[i] = defaultdict(list)   
        self.test_results_same, self.test_results_diff = defaultdict(list), defaultdict(list)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255,reduction = 'none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        
        for cluster_id in range(len(self.style_augment.cluster_mapping)):


            print("Starting training for cluster / model: ",cluster_id)
            self.model.load_state_dict(self.initial_state_dict)
            self.train_loader.dataset.style_augment.current_cluster = cluster_id

            self.get_optimizer()
            self.get_scheduler()

            for epoch in range(self.args.num_epochs):
                self.train_epoch(cluster_id)           
            self.states_dict[cluster_id] = copy.deepcopy(self.model.state_dict())

        self.test(dom = "same")
        self.test(dom = "diff")
        self.print_results()
                
    def generate_datasets(self):

        print(' Generating datasets...')
        train_dataset, valid_datasets, test_datasets = get_datasets(self.args)
        self.client_datasets = valid_datasets if self.args.style_transfer else None
        valid_dataset = ConcatDataset(valid_datasets)
        print(' ...done')

        self.valid_loader = DataLoader(valid_dataset, batch_size=self.args.bs, shuffle=False, num_workers=self.args.nw)
        if self.args.test:
            self.test_loader_samedom = DataLoader(test_datasets[0], batch_size=1, shuffle=False, num_workers=self.args.nw)
            self.test_loader_diffdom = DataLoader(test_datasets[1], batch_size=1, shuffle=False, num_workers=self.args.nw)
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
        elif self.args.policy == 'None':
            self.scheduler = None

    def train_epoch(self,cluster_id):

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
        self.train_results[cluster_id]['loss'].append(train_loss)
        self.train_results[cluster_id]['mIoU'].append(metric_results['Mean IoU'])
        self.train_results[cluster_id]['fwIoU'].append(metric_results['FreqW Acc'])
        self.train_results[cluster_id]['pAcc'].append(metric_results['Mean Acc'])

    def print_train_stats(self):
        print("\n Train results for current epoch: ")
        print("\t loss: ", self.train_results['loss'][-1])
        print("\t mIoU: ", self.train_results['mIoU'][-1])
        print("\t fwIoU: ", self.train_results['fwIoU'][-1])
        print("\t pAcc: ", self.train_results['pAcc'][-1])

    def test(self, dom):

        loader = self.test_loader_samedom if dom == 'same' else self.test_loader_diffdom
        mode = "Test same dom" if dom=="same" else "Test diff dom"

        aggregated_outputs, aggregated_predictions = dict(), dict()

        self.metric.reset()
        metric_dict, loss_dict = self.create_dicts(copy.deepcopy(self.metric))
        
        self.model.eval()
        with torch.no_grad():
            
            for i, (image, label) in enumerate(loader):
                
                #1. Extract the style of the image and flat the style
                style = self.style_augment.extract_style_for_test(image)
                style_flat = style.reshape(-1)
                
                #2. Check the closeness to each cluster
                distances = self.style_augment.best_clustering.transform([style_flat])
                similarities = 1.0 / (1.0+distances)        
                similarities = np.divide(similarities,np.sum(similarities)) 

                #3. For each model compute the prediction, then aggregate for each aggregation type

                image, label = image.to("cuda:0", dtype=torch.float32), label.to("cuda:0", dtype=torch.int64)

                outputs = []
                for cluster_id, state_dict in enumerate(self.states_dict.values()):
                    self.model.load_state_dict(state_dict)
                    model_output = self.model(image)["out"]
                    outputs.append(model_output)

                similarities = similarities.squeeze()
                aggregated_outputs[1], aggregated_predictions[1] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='max')
                aggregated_outputs[2], aggregated_predictions[2] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='mean', weighting_scheme='standard')
                aggregated_outputs[3], aggregated_predictions[3] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='mean', weighting_scheme='skewed')
                aggregated_outputs[4], aggregated_predictions[4] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='median', weighting_scheme='standard')
                aggregated_outputs[5], aggregated_predictions[5] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='median', weighting_scheme='skewed')
                aggregated_outputs[6], aggregated_predictions[6] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='majority')
                aggregated_outputs[7], aggregated_predictions[7] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='random_by_output', weighting_scheme='standard')
                aggregated_outputs[8], aggregated_predictions[8] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='random_by_output', weighting_scheme='skewed')
                aggregated_outputs[9], aggregated_predictions[9] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='random_by_pixel', weighting_scheme='standard')
                aggregated_outputs[10], aggregated_predictions[10] = self.aggregate_outputs(outputs=outputs, similarities=similarities, aggregation='random_by_pixel', weighting_scheme='skewed')

                for i, k in enumerate(metric_dict.keys(),start=1):

                    #4. Compute the loss
                    if aggregated_outputs[i] != None:
                        loss = self.reduction(self.criterion(aggregated_outputs[i], label),label)
                        loss_dict[k] += loss.item()
                        
                    #5. Update the metric  
                    metric_dict[k].update(label.detach().cpu().numpy(), aggregated_predictions[i].detach().cpu().numpy())
                
        print(f"\n {mode} results:")
        for i, k in enumerate(metric_dict.keys(),start=1):
            print(f"\n Aggregation: {k}")
            if aggregated_outputs[i] != None:
                loss_dict[k] /= len(loader)
                print(f"\t loss: {loss_dict[k]}")
            else:
                print(f"\t loss: n/a")
            metric_results = metric_dict[k].get_results()
            print("\t mIoU: ", metric_results['Mean IoU'])
            print("\t fwIoU: ", metric_results['FreqW Acc'])
            print("\t pAcc: ", metric_results['Mean Acc'])

    def create_dicts(self,metric):
        metric_dict, loss_dict = dict(), dict()
        keys = ['max', 'mean_standard', 'mean_skewed', 'median_standard', 'median_skewed', 'majority', 'randOutput_standard', 'randOutput_skewed',
                'randPixel_standard', 'randPixel_skewed']
        for key in keys:
            metric_dict[key] = copy.deepcopy(metric)
            loss_dict[key] = 0.0
        return metric_dict, loss_dict
                
    def aggregate_outputs(self, outputs, similarities, aggregation, weighting_scheme=None):

        if weighting_scheme != None:
            if weighting_scheme == 'standard':   
                weighted_outputs = [output * similarity for output,similarity in zip(outputs,similarities)]
            elif weighting_scheme == 'skewed':              
                similarities = np.power(similarities, 3)       
                similarities /= np.sum(similarities)  
                weighted_outputs = [output * similarity for output,similarity in zip(outputs,similarities)]
        
        #1. max
        if aggregation=='max':
            max_similarity_idx = np.argmax(similarities)
            output = outputs[max_similarity_idx]
            predicted_label = torch.argmax(output, dim=1) 
            return output, predicted_label
        
        #2. Mean
        elif aggregation == 'mean':
            output = torch.mean(torch.stack(weighted_outputs), dim = 0)
            predicted_label = torch.argmax(output, dim=1) 
            return output, predicted_label

        #3. Median
        elif aggregation == 'median':
            output = torch.median(torch.stack(weighted_outputs), dim = 0)[0]
            predicted_label = torch.argmax(output, dim=1) 
            return output, predicted_label
        
        #4. Voting
        elif aggregation == 'majority':
            class_predictions = [torch.argmax(output,dim=1) for output in outputs]
            predicted_label, _ = torch.mode(torch.stack(class_predictions), dim=0)
            return None, predicted_label
        
        #5. Random by output
        elif aggregation == 'random_by_output':
            #We are choosing the output of the models with probabilities given by the similarities.
            # E.g., with prob weighted_similarities[0] it will choose outputs[0] and so on.... 
            probabilities = torch.tensor(similarities).to("cuda:0",dtype=torch.float32)
            chosen_index = torch.multinomial(probabilities, 1).item()
            output = outputs[chosen_index]
            predicted_label = torch.argmax(output, dim=1) 
            return output, predicted_label
        
        #6. Random by pixel
        elif aggregation == 'random_by_pixel':
            predicted_labels = [torch.argmax(output,dim=1).squeeze() for output in outputs]
            stacked_labels = torch.stack(predicted_labels, dim=0)
            choices = np.arange(len(similarities))
            indices = np.random.choice(choices, size=(1080, 1920), p=similarities)
            final_label =  torch.zeros(1080, 1920)
            i_index, j_index = np.ogrid[:indices.shape[0], :indices.shape[1]]
            final_label = stacked_labels[indices, i_index, j_index]
            final_label = final_label.unsqueeze(0)
            return None,final_label

    def print_results(self):

        #### Plotting Results (train) ####

        metric_names = ['loss', 'mIoU','fwIoU','pAcc']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, metric in enumerate(metric_names):
            ax = axes[i]
            for cluster_id in self.train_results.keys():
                ax.plot(np.arange(1, self.args.num_epochs+1), self.train_results[cluster_id][metric], label='Model C'+str(cluster_id))
            ax.set_title("Train "+metric)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric)
            num_ticks = self.args.num_epochs//10 
            ax.set_xticks(list(range(self.args.num_epochs)), minor=True)
            ax.set_xticks(list(range(self.args.num_epochs))[::num_ticks])
            ax.legend()
            ax.grid()
        fig.suptitle(f" T5 Exp. {self.args.n_exp}")
        plt.tight_layout()
        os.makedirs(os.path.join('Results','Plots','Plots_T5'), exist_ok=True)
        plt.savefig(os.path.join('Results','Plots','Plots_T5', f'exp_{self.args.n_exp}.png'))
        plt.close(fig)
            



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
    parser.add_argument('--policy', type=str, choices=['None', 'step', 'poly'], help='scheduler')
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

def experiment_t5_ext1(args_dict):
    #create the parser
    args = get_parser(args_dict)
    print_setup(args)
    #set the seed
    set_seed(args.seed)
    #start the experiment
    trainer = Trainer(args)
    trainer.train()