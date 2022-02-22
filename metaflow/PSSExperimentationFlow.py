# https://github.com/valayDave/imagenet-with-metaflow

from metaflow import FlowSpec, step, Parameter
import re

def get_docid_radical_and_page(row):
    match = re.match(r"^([a-zA-Z0-9\-]*)(_(\d+))?$",row['docid'])
    if match:
        row['radical'] = match.groups()[0]
        row['page'] = int(match.groups()[2]) if match.groups()[2] else 1
    else:
        print('error', row)
    return row

def get_extended_class(x):
    page = x['page'] 
    pages = x['pages']
    if pages == 1: 
        x['extended_class'] = 'single page'
        return x
    if page == 1:
        x['extended_class'] = 'first of many'
        return x
    if page == pages:
        x['extended_class'] = 'last page'
        return x
    x['extended_class'] = 'middle'
    return x

def add_extended_class_column(df):
    df = df.apply(get_docid_radical_and_page, axis=1)
    df_aux = df.groupby(['radical'], as_index=False)[['page']].max()
    df_aux.rename(columns={'page':'pages'}, inplace=True)
    df = df.merge(df_aux, how='left', on='radical')
    df = df.apply(get_extended_class, axis=1)
    return df.copy()

class PSSExperimentationFlow(FlowSpec):
    
    workers = Parameter('workers', default=3, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    epochs = Parameter('epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    start_epoch = Parameter('start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    batch_size = Parameter('batch_size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    learning_rate = Parameter('learning_rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    momentum = Parameter('momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    weight_decay = Parameter('wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        )
    print_frequency = Parameter('pf', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')    
    
    seed = Parameter('seed', default=1234, type=int,
                    help='seed for initializing training. ')
    
    
    
    @step
    def start(self):
        import pandas as pd
        
        self.trained_on_gpu = False
        self.used_num_gpus = 0 
        
        classes = [2,4]
        inputs = [1,2,3]
        models = ['VGG16', 'Eff']
        
        
        self.training_models = [{'nn':model, 'input': inp,  'output':cls}\
                        for model in models\
                        for inp in inputs\
                        for cls in classes ]
      #  self.training_models = [{'nn': 'VGG16', 'input': 2, 'output': 2}]
    
    
        self.PAGE_IMGS_PATH = '/mnt/nas/databases/Tobacco800/unziped/page_imgs/raw/'
        self.TRAIN_LABEL_PATH = '/mnt/nas/databases/Tobacco800/unziped/train.csv'
        self.TEST_LABEL_PATH = '/mnt/nas/databases/Tobacco800/unziped/test.csv'
        
        
        self.df_train = pd.read_csv(self.TRAIN_LABEL_PATH, sep=';', skiprows=0, low_memory=False)
        self.df_test = pd.read_csv(self.TEST_LABEL_PATH,sep=';', skiprows=0, low_memory=False) 
        
        #print(self.df_test.shape)
        
        self.df_val = self.df_train.iloc[:200,:]
        self.df_train = self.df_train.iloc[200:,:]
        
        self.df_train = add_extended_class_column(self.df_train)
        self.df_val = add_extended_class_column(self.df_val)
        self.df_test = add_extended_class_column(self.df_test)
        
        print(f'Train: {len(self.df_train)} Valid: {len(self.df_val)} Test: {len(self.df_test)}')        

        self.next(self.train_model,foreach='training_models')

    @step
    def train_model(self):
        import PSS_pytorch
        import json
        
        self.arch = self.input
        
        neural_net = self.arch['nn']
        inp = self.arch['input']
        output = self.arch['output']
        
        print(f'{neural_net} page_{inp} class_{output}')
        
        self.evaluate = False
        
        results, model = PSS_pytorch.start_training_session(self)
        #model = model.to('cpu') # Save CPU based model state dict. 
        
        self.model = model
        
        self.epoch_histories = json.loads(json.dumps(results))
        
        self.evaluate = True
 
        results, test_preds, test_targets = PSS_pytorch.start_training_session(self)
    
        self.test_result = results
        
        self.test_targets = test_targets        
        
        self.test_preds = test_preds

        self.next(self.join)
        
    @step
    def join(self,inputs):
        
        from reporting_data import ModelAnalytics,FinalModel
        self.history = []
        self.models = []
        for input_val in inputs:
            
            neural_net = input_val.arch['nn']
            inp = input_val.arch['input']
            output = input_val.arch['output']

            arch = f'{neural_net} page_{inp} class_{output}'            
            
            print("Downloading Models/Data for Arch,",arch)
            model_results = ModelAnalytics()
            final_model = FinalModel()
            # Saving Analytics Results. 
            model_results.architecture = arch
            model_results.epoch_histories = input_val.epoch_histories
            model_results.hyper_params.batch_size = self.batch_size
            model_results.hyper_params.momentum = self.momentum
            model_results.hyper_params.weight_decay = self.weight_decay
            model_results.num_gpus = input_val.used_num_gpus
            model_results.test_result = input_val.test_result
            model_results.test_preds = input_val.test_preds
            model_results.test_targets = input_val.test_targets
            # Saving Model Results. 
            final_model.architecture = model_results.architecture
            final_model.model = input_val.model
            input_val.model = None
            final_model.hyper_params = model_results.hyper_params
            final_model.epochs = len(input_val.epoch_histories['train'])
            self.models.append(final_model)
            self.history.append(model_results)   

        self.next(self.end)
    @step
    def end(self):
        print("Completed Computation !")
        # self.next(self.last)

if __name__ == '__main__':
    PSSExperimentationFlow()
