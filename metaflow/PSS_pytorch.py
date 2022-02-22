import random
import torch as T
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import importlib
import timm
from timm.optim import optim_factory
from types import SimpleNamespace

from torchsummary import summary

import time
import warnings

best_acc = 0

def start_training_session(parsed_arguements):
    state_object = parsed_arguements
    if state_object.seed is not None:
        random.seed(state_object.seed)
        T.manual_seed(state_object.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    return main_worker(state_object)


def main_worker(state_object):
    global best_acc
    
    neural_net = state_object.arch['nn']
    inp = state_object.arch['input']
    output = state_object.arch['output']
    model_class = f'{neural_net}_{inp}'
    
    arch = f'{neural_net} page_{inp} class_{output}'    
    

    Model = getattr(importlib.import_module("nn_modules"), model_class)
    if neural_net == 'Eff':
        model = Model(output, False)
        lr = 5e-3

        params = [
                  {'params': model.base_model.parameters(), 'lr': 5e-3 / 10},
                  {'params': model.classifier.parameters()}
                 ]
        optimizer = optim.Adam(params, lr = lr)        
    else:
        model = Model(output)    
        args = SimpleNamespace()
        args.weight_decay = 0 
        args.lr = 5e-5
        args.opt="Nadam"
        optimizer = optim_factory.create_optimizer(args, model)
    
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    print("Going to Train on ","cuda" if T.cuda.is_available() else "cpu")
    state_object.trained_on_gpu = True if T.cuda.is_available() else False
    
    state_object.used_num_gpus = T.cuda.device_count()
#     if T.cuda.device_count() > 1:
#         print("Let's use", T.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model)
    
    train_with_gpu = False
    if T.cuda.is_available():
        train_with_gpu = True
                
    model.to(device)
        
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
#     optimizer = T.optim.SGD(model.parameters(), state_object.learning_rate,
#                                 momentum=state_object.momentum,
#                                 weight_decay=state_object.weight_decay)
    
    
    Dataset = getattr(importlib.import_module("dataset_modules"), f'TobaccoDataset{inp}')

    ResizeBinarize = getattr(importlib.import_module("dataset_modules"), f'ResizeBinarize{neural_net}')
    
    transform = transforms.Compose([
                            ResizeBinarize(),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: T.cat([x, x, x], 0))
                       ])
    
    if output == 2:
        label2Idx = {'FirstPage' : 1, 'NextPage' : 0}
        class_column = 'class'
    else:
        label2Idx = {'single page': 3,'first of many': 2,'middle': 1,'last page':0}
        class_column = 'extended_class'

    cudnn.benchmark = True
    
    
    if state_object.evaluate:
        model.load_state_dict(state_object.model)
        test_data = Dataset(state_object.df_test, label2Idx, class_column, state_object.PAGE_IMGS_PATH, transform )
        test_loader = data.DataLoader(test_data, 
                                     batch_size = state_object.batch_size, shuffle=False)
    
        
        loss, acc, test_history, test_preds, test_targets = validate(test_loader, model, criterion, state_object, device)
        return test_history, test_preds, test_targets
    
    train_data = Dataset(state_object.df_train, label2Idx, class_column, state_object.PAGE_IMGS_PATH, transform )

    valid_data = Dataset(state_object.df_val, label2Idx, class_column, state_object.PAGE_IMGS_PATH, transform )    
    
    train_loader = data.DataLoader(train_data, 
                                     batch_size = state_object.batch_size, shuffle=False)

    val_loader = data.DataLoader(valid_data, 
                                     batch_size = state_object.batch_size, shuffle=False)

    
    
    print("Training/Testing Datasets Loaded!")
    epoch_histories = {
        'train': [],
        'validation': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(state_object.start_epoch, state_object.epochs):
        #adjust_learning_rate(optimizer, epoch, state_object)
        if epoch % 2 == 0:
            print("Training Epoch : ",epoch)
        # train for one epoch
        train_history = train(train_loader, model, criterion, optimizer, epoch, state_object, device)
        epoch_histories['train'].append(train_history)
        # evaluate on validation set
        loss, acc, validation_history, _, _ = validate(val_loader, model, criterion, state_object, device)
        epoch_histories['validation'].append(validation_history)
        # remember best acc@1 and save checkpoint
        if best_loss > loss:
            loss = best_loss
            print(f'\t\t### Best Model for {arch}\n')
            model_params = model.state_dict()

    return epoch_histories , model_params

def train(train_loader, model, criterion, optimizer, epoch, state_object, device):
    
    input_size = state_object.arch['input']
    output_size = state_object.arch['output']
    
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[]
    }
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top_acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top_acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    # https://github.com/pytorch/pytorch/issues/16417#issuecomment-566654504
    end = time.time()
    for i, (images, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        optimizer.zero_grad()
                
        if input_size == 1:
            images = images.to(device,non_blocking=True)
            # compute output
            output = model(images) 
            
        elif input_size == 2:
            x1, x2 = images

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            
            # compute output
            output = model(x1, x2) 
            
        elif input_size == 3:
            x1, x2, x3 = images

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            x3 = x3.to(device, non_blocking=True)
            
            # compute output
            output = model(x1, x2, x3)  
            
        target = target.to(device, non_blocking=True)
            
        loss = criterion(output, target)

        # measure accuracy and record loss
        
        output = output.argmax(1, keepdim = True)
        
        if output_size == 4:
            target = target//2
            output = output//2
            
        #print(name)
        
        acc = accuracy(output, target)
        losses.update(loss.item(), target.size(0))
        top_acc.update(acc, target.size(0))


        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        history['accuracy'].append(float(top_acc.avg))
        history['loss'].append(float(losses.avg))
        history['batch_time'].append(float(batch_time.avg))
        
        if i % state_object.print_frequency == 0:
            progress.display(i)
    
    return history

def validate(val_loader, model, criterion, state_object,device):
    
    input_size = state_object.arch['input']
    output_size = state_object.arch['output']
    preds = []
    targets = []
    
    history = {
        'loss': [],
        'accuracy':[],
        'batch_time':[]
    }
    
    prefix = 'Test: ' if state_object.evaluate else 'Valid: '
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    top_acc = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top_acc],
        prefix=prefix)

    # switch to evaluate mode
    model.eval()

    with T.no_grad():
        end = time.time()
        for i, (images, target, _) in enumerate(val_loader):
            if input_size == 1:
                images = images.to(device,non_blocking=True)
                
                # compute output
                output = model(images)
            elif input_size == 2:
                x1, x2 = images

                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)

                # compute output
                output = model(x1, x2)

            elif input_size == 3:
                x1, x2, x3 = images

                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                x3 = x3.to(device, non_blocking=True)

                # compute output 
                output = model(x1, x2, x3)
                
                
            
            target = target.to(device,non_blocking=True)

            loss = criterion(output, target)

            # measure accuracy and record loss
            
            #print(output.tolist())
            
            output = output.argmax(1, keepdim = True)
            
            if output_size == 4:
                target = target//2
                output = output//2
            
            preds = preds + (output.squeeze().tolist())
            targets = targets + (target.squeeze().tolist())
            
            acc = accuracy(output, target)
            #print('acc ', acc)
            losses.update(loss.item(), target.size(0))
            top_acc.update(acc, target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            history['accuracy'].append(float(top_acc.avg))
            history['loss'].append(float(losses.avg))
            history['batch_time'].append(float(batch_time.avg))

            if i % state_object.print_frequency == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc {top_acc.avg:.3f}'
              .format(top_acc=top_acc))

    return losses.avg, top_acc.avg, history, preds, targets

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    T.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, state_object):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = state_object.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target): 
    correct = output.eq(target.view_as(output)).sum()
    acc = correct.float() / target.shape[0]
    return acc