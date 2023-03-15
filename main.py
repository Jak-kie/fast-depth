import os
import time
import csv
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"                 # MODIFICA NOSTRA

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils


args = utils.parse_command()
# print(args)
print("args: " , args)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"              # WIP, Usato per settare la CPU al posto della GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu         # Set the GPU.  ORIGINALE

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # Data loading code
    print("=> creating data loaders...")
    valdir = os.path.join('..', 'data', args.data, 'val')

    if args.data == 'nyudepthv2':
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.')

    # set batch size to be 1 for validation
    """num_workers causa BrokenPipeError: [Errno 32] Broken pipe, cio è dovuto alla mancanza di RAM (o VRAM, ma dubito avendone solo 2 GB).
        Chiudere piu app possibili prima di avviare il programma"""
    """il numero di workers usati indica il numero di output di namespace; probabile che indichi il
        numero di processi paralleli creati"""
    # args.workers è 16, cambiarlo non sembra impattare le performance, necessario misurare il tempo precisamente tho
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)   # ORIGINALE
    print("=> data loaders created.")

    check_is_cuda_used()

    # evaluation mode
    if args.evaluate:
        assert os.path.isfile(args.evaluate), \
        "=> no model found at '{}'".format(args.evaluate)
        print("=> loading model '{}'".format(args.evaluate))        
        # checkpoint = torch.load(args.evaluate)        # ORIGINALE
        checkpoint = torch.load(args.evaluate, map_location=torch.device('cpu'))
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


# flashing della memoria, per evitare errori di runtime
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


def validate(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()
    """
    Viene eseguita una iterazione per ogni immagine valutata (in questo caso le 654 immagini di /val)
    """
    for i, (input, target) in enumerate(val_loader):
        # input, target = input.cuda(), target.cuda()       # ORIGINALE
        input, target = input.cpu() , target.cpu()
        # print ("OUTPUT PROVA 1")                          # durante il testing di CPU, qua si arrestava
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
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
