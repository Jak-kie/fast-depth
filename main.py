import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils

import matplotlib.pyplot as plt         # per validateSingleImage 
from skimage.transform import resize


# COMANDO python .\main.py --evaluate ..\results\mobilenet-nnconv5dw-skipadd-pruned.pth.tar --device 0


args = utils.parse_command()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device        # Setta il device usato
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu         # Set the GPU.  ORIGINALE

# rimpiazzare gpu_time con device_time da problemi con la libreria metrics,
# quindi cambiamolo solo quando è in output
fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()


def main():
    global args, best_result, output_directory, train_csv, test_csv

    torch.cuda.empty_cache()

    check_is_cuda_used()

    # richiama l'analisi della singola immagine
    if True:
        validateSingleImage()
        return

    # Data loading code
    print("=> creating data loaders...")
    valdir = os.path.join('..', 'data', args.data, 'val')

    if args.data == 'nyudepthv2':
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("=> data loaders created.")

    # evaluation mode
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))        
        # checkpoint = torch.load(args.evaluate)        # ORIGINALE
        if (args.device == "-1"):           # carica CPU
            checkpoint = torch.load(args.evaluate, map_location=torch.device('cpu'))
        else:                               # carica GPU
            checkpoint = torch.load(args.evaluate)            
        if type(checkpoint) is dict:
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
        else:
            model = checkpoint
            args.start_epoch = 0
        output_directory = os.path.dirname(args.evaluate)
        validate(val_loader, model, args.start_epoch, write_to_file=False)
        return


# controllo per vedere se/quale GPU è usata
def check_is_cuda_used():
    print("torch.__version__ :" , torch.__version__)
    print("torch.cuda.is_available(): " , torch.cuda.is_available())
    print("torch.cuda.device_count(): " , torch.cuda.device_count())
    if (torch.cuda.is_available()):  # in caso si usi la CPU
        print("torch.cuda.current_device(): " , torch.cuda.current_device())
        print("torch.cuda.device(???)" , torch.cuda.device(torch.cuda.current_device()))
        print("torch.cuda.get_device_name(???)" , torch.cuda.get_device_name(torch.cuda.current_device()))


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    # Viene eseguita una iterazione per ogni immagine valutata (in questo caso le 654 immagini di /val)
    for i, (input, target) in enumerate(val_loader):
        if (args.device == "-1"):
            input, target = input.cpu() , target.cpu()
        else:
            input, target = input.cuda(), target.cuda()     # ORIGINALE
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50

        if args.modality == 'rgb':
            rgb = input

        """
        rgb = immagine di input iniziale
        target = truth mask
        pred = prediction del modello 
        """
        if i == 0:
            img_merge = utils.merge_into_row(rgb, target, pred)
        elif (i < 8*skip) and (i % skip == 0):
            row = utils.merge_into_row(rgb, target, pred)
            img_merge = utils.add_row(img_merge, row)
        elif i == 8*skip:
            filename = output_directory + '/comparison_' + str(epoch) + '.png'
            utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_DEVICE={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    # modificato t_GPU --> t_DEVICE
    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_DEVICE={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    # TODO vedere se gpu_time puo essere sostituito con device_time
    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'device_time': avg.gpu_time})
    return avg, img_merge


"""
INPUT: 1 immagine (path oppure file immagine)
OUTPUT: depth map in formato [224, 224], con valori [0,1]
"""
def validateSingleImage():
    
    outputSize = (224, 224)

    imagePath = "jj.jpg"

    # temporaneamente usiamo i parametri per ottenere la immagine
    # valdir = os.path.join('..', 'data', args.data, 'val', 'official', '00001.h5')
    
    # N.B. /255. suppone immagini RGB a 8 bit
    inputImage = plt.imread(imagePath)/255.    # normalization, valori dei pixel tra [0,1]

    inputImage = resize(inputImage, outputSize)                 # (224, 224, 3)

    # usa np.transpose al posto di np.reshape, risultati decisamente migliori
    # inputImage = np.reshape(inputImage, (3, 224, 224))            # (3, 224, 224)
    inputImage = np.transpose(inputImage, (2,0,1))              # (3, 224, 224)
    inputImage = np.expand_dims(inputImage, axis=0)               # (1, 3, 224, 224)

    # per funzionare il model richiede una immagine in formato NCHW [1,3,224,224]

    # carica il modello passato per args
    if (args.device == "-1"):           # carica CPU
        checkpoint = torch.load(args.evaluate, map_location=torch.device('cpu'))
        inputImage = torch.from_numpy(inputImage).float().cpu()
    else:                               # carica GPU
        checkpoint = torch.load(args.evaluate)
        inputImage = torch.from_numpy(inputImage).float().cuda()       
    if type(checkpoint) is dict:
        args.start_epoch = checkpoint['epoch']
        model = checkpoint['model']
    else:
        model = checkpoint
        args.start_epoch = 0

    # effettua la prediction
    with torch.no_grad():
        pred = model(inputImage)
    
    """
    output tipo di pred
    tensor([[[[1.7303, 1.7303, 1.7306,  ..., 1.6876, 1.7618, 1.7618],
          [1.7303, 1.7303, 1.7306,  ..., 1.6876, 1.7618, 1.7618],
          [1.6839, 1.6839, 1.6673,  ..., 1.6338, 1.6839, 1.6839],
          ...,
          [1.3466, 1.3466, 1.3264,  ..., 1.8757, 1.8533, 1.8533],
          [1.4089, 1.4089, 1.3693,  ..., 1.7751, 1.7419, 1.7419],
          [1.4089, 1.4089, 1.3693,  ..., 1.7751, 1.7419, 1.7419]]]],
       device='cuda:0')
    """

    """
    pred.size()         -->  torch.Size([1, 1, 224, 224])
    torch.squeeze(pred) -->  torch.Size([224, 224])
    """

    """
    .numpy() garantisce una precisione maggiore, oltre a renderlo un array .numpy       1.7751 --> 1.7751079
        valutare se serve che sia un numpyarray
    .cpu() rimuove l'eventuale "device='cuda:0'", non penso faccia altro

    pred1 = torch.squeeze(pred)
    pred2 = np.squeeze(pred.data.cpu().numpy())

    plt.imshow(pred1)
    plt.show()
    plt.imshow(pred2)
    plt.show()

    print(pred1.data.cpu().numpy())
    print(pred2)

    print(pred1.shape)
    print(pred2.shape)
    """

    pred = torch.squeeze(pred)
    # ora pred è una matrice 2D, con valori validi di depth map
    plt.imshow(pred)
    plt.show()

    minPred = torch.min(pred)
    maxPred = torch.max(pred)
    print("Valore minimo di pred: " , minPred)
    print("Valore MASSIMO di pred: " , maxPred)


if __name__ == '__main__':
    main()
