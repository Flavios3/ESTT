import argparse


def get_parser(args_dict):
    parser = argparse.ArgumentParser()
    #arguments for the experimental setup
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset', type=str, choices=['idda', 'gtaV'], help='dataset name')
    parser.add_argument('--model', type=str, choices=['deeplabv3_mobilenetv2', 'resnet18', 'cnn'], help='model name')
    parser.add_argument('--num_rounds', type=int, help='number of rounds')
    parser.add_argument('--num_epochs', type=int, help='number of local epochs')
    parser.add_argument('--n_exp', type=int, help='number of the experiment')
    parser.add_argument('--clients_per_round', type=int, help='number of clients trained per round')
    parser.add_argument('--print_times', type=bool, default=False, help='whether to print the time needed for a train/eval loop')
    parser.add_argument('--target_miou', type=float, default=0.0, help='used to run until target miou is reached instead of running for num_epochs')
    #arguments for the model parameters
    parser.add_argument('--da', type=str, choices=['basic', 'advanced'], help='type of data augmentation')
    parser.add_argument('--bs', type=int, default=4, help='batch size')
    parser.add_argument('--nw', type=int, default=2, help='num workers')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--wd', type=float, default=0, help='weight decay')
    #arguments for the criterion, optimizer and scheduler
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hard negative mining reduction or not')
    parser.add_argument('--opt', type=str, choices=['None', 'SGD', 'Adam'], help='optimizer')
    parser.add_argument('--m', type=float, default=0.9, help='momentum')
    parser.add_argument('--Adam_p', type=str, choices=['dft', 'fd', 'mid'], help='set Adams parameter to the default or FedDrive ones, or try a middle ground')
    parser.add_argument('--policy', type=str, choices=['None', 'step', 'poly'], help='scheduler')
    parser.add_argument('--lr_power', type=float, default=0.9, help='poly scheduler power')
    parser.add_argument('--lr_step', type=int, default=15, help='step scheduler decay step')
    parser.add_argument('--lr_factor', type=float, default=0.1, help='step scheduler decay factor')
    #arguments for the handling of validation and testing
    parser.add_argument('--test', type=bool, default=True, help='whether to test')
    parser.add_argument('--warmup', type=float, default=0.5, help='percentage of the total epochs/rounds to be considered as warmup (only training, no validation)')
    parser.add_argument('--train_display_interval', type=int, default=5, help='epoch/round interval for the display of the train statistics')
    parser.add_argument('--valid_interval', type=int, default=5, help='epoch/round interval for a validation loop')
    parser.add_argument('--n_final_rounds', type=int, default=5, help='number of final rounds, on which to densily evaluate')
    parser.add_argument('--save_best_model', type=bool, default=True, help='whether to save the best model among the last final_rounds')
    #arguments for handling SSL task
    parser.add_argument('--mode', type=str, choices=['SL', 'SSL'], help='type of learning paradigm')
    parser.add_argument('--teacher_strat', type=int, default = 1,choices=[1,2,3], help='training strategy for the teacher')
    parser.add_argument('--T', type=int, default = 5, help='T parameter for teacher strategy == 3')
    parser.add_argument('--path_model_SSL',type=str,help='path to pretrained model to load in SSL task either model from T3.2 or T3.4')

    args = parser.parse_args(args=[])
    for k, v in args_dict.items():
        setattr(args, k, v)
    
    return args