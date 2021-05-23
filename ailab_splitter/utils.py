from sklearn.metrics import classification_report
import json
from timm import create_model
from fastai.vision.all import *
from matplotlib import pyplot as plt
import numpy as np

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_num_params(model, display_all_modules=False):
    total_num_params = 0
    for n, p in model.named_parameters():
        num_params = 1
        for s in p.shape:
            num_params *= s
        if display_all_modules: print("{}: {}".format(n, num_params))
        total_num_params += num_params
    print("-" * 50)
    print("Total number of parameters: {:.2e}".format(total_num_params))

def save_plot_loss_curves(experiment, loss_values):
    train_loss_values, valid_loss_values = loss_values
    #Plot the loss curves
    plt.figure(figsize=[8,6])
    plt.plot(train_loss_values,'r',linewidth=3.0)
    plt.plot(valid_loss_values,'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=1)
    plt.savefig('./model/'+ experiment+ '_loss_curves.png')

def save_plot_acc_curves(experiment, loss_values):
    #Plot the Accuracy Curves
    plt.figure(figsize=[8,6]) 
    plt.plot(train_acc_values,'r',linewidth=3.0) 
    plt.plot(valid_acc_values,'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16) 
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.savefig('./model/'+ experiment+ '_acc_curves.png')
    
def plot_confusion_matrix(labels, pred_labels, experiment, classes):    
    fig = plt.figure(figsize = (10, 10));
    ax = fig.add_subplot(1, 1, 1);
    cm = confusion_matrix(labels, pred_labels);
    cm = ConfusionMatrixDisplay(cm, display_labels = classes);
    cm.plot(values_format = 'd', cmap = 'Blues', ax = ax)
    plt.xticks(rotation = 20)
    plt.savefig('./model/'+ experiment+ '_confusion_matrix.png')
    
    
def create_timm_body(arch:str, pretrained=True, cut=None):
    model = create_model(arch, pretrained=pretrained)
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int): return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut): return cut(model)
    else: raise NamedError("cut must be either integer or function")
    
def concat_two_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_three_images(imga, imgb, imgc):
    """
    Combines three color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    hc,wc = imgc.shape[:2]
    max_height = np.max([ha, hb, hc])
    total_width = wa+wb+wc
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    new_img[:hc,wa+wb:wa+wb+wc]=imgc
    return new_img

def normalize_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min = image_min, max = image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def plot_three_images(images, labels, classes, file_names, normalize = True):

    n_images = len(labels)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):
        
        prev_image = images[i][0]
        curr_image = images[i][1]
        next_page = images[i][2]

        ax = fig.add_subplot(rows, cols, i+1)
        
        if normalize:
            prev_image = normalize_image(prev_image)
            curr_image = normalize_image(curr_image)
            next_page = normalize_image(next_page)

        prev_image = prev_image.permute(1, 2, 0).cpu().numpy()    
        curr_image = curr_image.permute(1, 2, 0).cpu().numpy()
        next_page = next_page.permute(1, 2, 0).cpu().numpy()

        ax.imshow(concat_three_images(prev_image, curr_image, next_page))
        ax.set_title(f'{classes[labels[i]]}\n' \
                     f'{file_names[i]}')   
        ax.axis('off')


def plot_two_images(images, labels, classes, file_names, normalize = True):

    n_images = len(labels)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (15, 15))

    for i in range(rows*cols):
        
        prev_image = images[i][0]
        curr_image = images[i][1]

        ax = fig.add_subplot(rows, cols, i+1)
        
        if normalize:
            prev_image = normalize_image(prev_image)
            curr_image = normalize_image(curr_image)

        prev_image = prev_image.permute(1, 2, 0).cpu().numpy()    
        curr_image = curr_image.permute(1, 2, 0).cpu().numpy()

        ax.imshow(concat_two_images(prev_image, curr_image))
        ax.set_title(f'{classes[labels[i]]}\n' \
                     f'{file_names[i]}')   
        ax.axis('off')

def plot_one_image(images, labels, classes, file_names, normalize = True):

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize = (45, 60))

    for i in range(rows*cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)
         
        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f'{classes[labels[i]]}\n' \
                     f'{file_names[i]}')   
#        ax.set_title(labels[i])
        ax.axis('off')


def export_report(experiment, 
                labels, 
                pred_labels, 
                target_names, 
                valid_loss, 
                valid_acc, 
                valid_kappa,
                test_loss, 
                test_acc, 
                test_kappa,                  
                report_file_path):
    '''
    Export experiment metrics to a file, merging with already saved reports.
    experimen        string experiment identification
    labels           list   ground truth
    pred_labels      list   predicted labels
    target_names     list   class names
    loss             float  loss metric 
    accuracy         float  accuracy 
    kappa            float  kappa 
    report_file_path string report file path
    '''
    report = {experiment: {}}
    report[experiment] = classification_report(labels, pred_labels, target_names=target_names, output_dict=True)
    report[experiment]['valid_acc'] = valid_acc
    report[experiment]['valid_kappa'] = valid_kappa
    report[experiment]['test_acc'] = test_acc
    report[experiment]['test_kappa'] = test_kappa
    print(report)
    
    try:
        with open(report_file_path) as json_file:
            data = json.load(json_file)
    except:
        print("An exception occurred when trying to open %s" % report_file_path)
        data = {}
    data = {**data, **report}
    with open(report_file_path, 'w') as outfile:
        json.dump(data, outfile)

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc