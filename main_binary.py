import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset, get_num_classes
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
import json
from torchvision.utils import save_image
import quantization 
from quantization.quant_auto import memory_driven_quant

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='vgg_cifar10_binary',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0,1,2,3',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='run model on validation set')
parser.add_argument('--save_check', action='store_true',
                    help='saving the checkpoint')
#binarization parameters
parser.add_argument('--quantizer', action='store_true',
                    help='using the quantizer flow')
parser.add_argument('--type_quant', default=None,
                    help='Type of binarization process')
parser.add_argument('--weight_bits', default=1,
                    help='Number of bits for the weights')
parser.add_argument('--activ_bits', default=1,
                    help='Number of bits for the activations')
parser.add_argument('--activ_type', default='hardtanh',
                    help='Type of the quantized activation layers')

parser.add_argument('--batch_fold_delay', default=0, type=int, 
                    help='Apply folding of batch layers into convolutional')
parser.add_argument('--batch_fold_type', default='folding_weights', type=str, 
                    help='Type of folding for batch norm layers: folding_weights | ICN')
parser.add_argument('--quant_add_config', default='', type=str, 
                    help='Additional config of per-layer quantization')

#mobilenet params
parser.add_argument('--mobilenet_width', default=1.0, type=float,
                    help='Mobilenet Width Muliplier')
parser.add_argument('--mobilenet_input', default=224, type=int,
                    help='Mobilenet input resolution ')

#mixed-precision params
parser.add_argument('--mem_constraint', default='', type=str,
                    help='Memory constraints for automatic bitwidth quantization')
parser.add_argument('--mixed_prec_quant', default='MixPL', type=str, 
                    help='Type of quantization for mixed-precision low bitwidth: MixPL | MixPC')


def main():
    global args, best_prec1 #全局变量
    best_prec1 = 0
    args = parser.parse_args()
    
    weight_bits = int(args.weight_bits) #量化参数
    activ_bits = int(args.activ_bits)


    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save) #保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        print('Selected GPUs: ', args.gpus)
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    nClasses = get_num_classes(args.dataset) #获取数据集分类数
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'num_classes': nClasses, \
                    'type_quant': args.type_quant, 'weight_bits': weight_bits, 'activ_bits': activ_bits,\
                    'activ_type': args.activ_type, 'width_mult': float(args.mobilenet_width), 'input_dim': float(args.mobilenet_input) }

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)
    print(model)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    print(transform)
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()    
    criterion.type(args.type)


    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.quantizer:
        val_quant_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=32, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


    #define optimizer
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        if 'clip_val' in key:
            params += [{'params':value,'weight_decay': 1e-4}]
        else:
            params += [{'params':value}]
    optimizer = torch.optim.SGD(params, lr=0.1)
    logging.info('training regime: %s', regime)

    #define quantizer 
    if args.quantizer:
        if args.mem_constraint is not '':
            mem_contraints = json.loads(args.mem_constraint)
            print('This is the memory constraint:', mem_contraints )
            if mem_contraints is not None:
                x_test = torch.Tensor(1,3,args.mobilenet_input,args.mobilenet_input)
                add_config = memory_driven_quant(model, x_test, mem_contraints[0], mem_contraints[1], args.mixed_prec_quant)
                if add_config == -1:
                    print('The quantization process failed!')
            else:
                add_config = []
        else:
            mem_constraint = None
            if args.quant_add_config is not '':
                add_config = json.loads(args.quant_add_config)

            else:
                add_config = []

        quantizer = quantization.QuantOp(model, args.type_quant, weight_bits, \
            batch_fold_type=args.batch_fold_type, batch_fold_delay=args.batch_fold_delay, act_bits=activ_bits, \
            add_config = add_config )
        quantizer.deployment_model.type(args.type)
        quantizer.add_params_to_optimizer(optimizer)

    else:
        quantizer = None

    #exit(0)


    #multi gpus
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.type(args.type)


    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint_loaded = torch.load(checkpoint_file)
            checkpoint = checkpoint_loaded['state_dict']
            model.load_state_dict(checkpoint, strict=False)
            print('Model pretrained')
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    if args.quantizer:
        quantizer.init_parameters()

    if args.evaluate:
        # evaluate on validation set

        if args.quantizer:
            # evaluate deployment model on validation set
            quantizer.generate_deployment_model()
            val_quant_loss, val_quant_prec1, val_quant_prec5 = validate(
                val_quant_loader, quantizer.deployment_model, criterion, 0, 'deployment' )
        else:
            val_quant_loss, val_quant_prec1, val_quant_prec5 = 0, 0, 0

        val_loss, val_prec1, val_prec5 = validate(
           val_loader, model, criterion, 0, quantizer)

        logging.info('\n This is the results from evaluation only: ' 
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'
                     'Validation Quant Prec@1 {val_quant_prec1:.3f} \t'
                     'Validation Quant Prec@5 {val_quant_prec5:.3f} \n'
                     .format(val_prec1=val_prec1, val_prec5=val_prec5,
                     val_quant_prec1=val_quant_prec1, val_quant_prec5=val_quant_prec5))
        exit(0)



    # 提前停止参数
    best_val_prec1 = 0
    patience = 30
    no_improve_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer, quantizer)
        #torch.cuda.empty_cache()


        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch, quantizer)
        torch.cuda.empty_cache()

        if args.quantizer:
            # evaluate deployment model on validation set
            quantizer.generate_deployment_model()
            val_quant_loss, val_quant_prec1, val_quant_prec5 = validate(
                val_quant_loader, quantizer.deployment_model, criterion, epoch, 'deployment' )
        else:
            val_quant_loss, val_quant_prec1, val_quant_prec5 = 0, 0, 0


        # 提前停止检查
        is_best = val_prec1 > best_val_prec1
        if is_best:
            best_val_prec1 = val_prec1
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1

        # 如果连续10个epoch没有改进，则停止训练
        if no_improve_epoch >= patience:
            print(f'Early stopping: no improvement for {patience} epochs')
            break

        # remember best prec@1 and save checkpoint
        best_prec1 = max(val_prec1, best_prec1)
          
     #save_model
        if args.save_check:

            print('Saving Model!! Accuracy : ', best_prec1)
            save_checkpoint({
                   'epoch': epoch + 1,
                   'model': args.model,
                   'config': model_config,
                   'state_dict': model.state_dict(),
                   'best_prec1': best_prec1,
                   'regime': regime ,
                   'quantizer': quantizer,
                   'add_config': add_config,
                   'fold_type': args.batch_fold_type
            }, is_best, path=save_path)


        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'
                     'Validation Quant Prec@1 {val_quant_prec1:.3f} \t'
                     'Validation Quant Prec@5 {val_quant_prec5:.3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5,
                             val_quant_prec1=val_quant_prec1, val_quant_prec5=val_quant_prec5))


        results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                    train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                    train_error5=100 - train_prec5, val_error5=100 - val_prec5,
                    val_quant_error1=100 - val_quant_prec1, val_quant_error5=100 - val_quant_prec5)
        results.save()
        # === 训练结束后导出 INT-only 部署模型 ===
    if args.quantizer:
        print('Exporting INT-only deployment model...')
        quantizer.generate_deployment_model()

        all_layers_params = []

        for i, layer in enumerate(quantizer.param_to_quantize):
            print(f"\n====== Layer {i} ======")

            quant_conv = layer.get('quant_conv', None)
            quant_act = layer.get('quant_act', None)

            if quant_conv is not None:
                conv_params = {
                    'weight': quant_conv.weight.detach().cpu().numpy() if hasattr(quant_conv, 'weight') else None,
                    'bias': quant_conv.bias.detach().cpu().numpy() if (hasattr(quant_conv, 'bias') and quant_conv.bias is not None) else None,
                    'in_channels': quant_conv.in_channels if hasattr(quant_conv, 'in_channels') else None,
                    'out_channels': quant_conv.out_channels if hasattr(quant_conv, 'out_channels') else None,
                    'kernel_size': quant_conv.kernel_size if hasattr(quant_conv, 'kernel_size') else None,
                    'stride': quant_conv.stride if hasattr(quant_conv, 'stride') else None,
                    'padding': quant_conv.padding if hasattr(quant_conv, 'padding') else None,
                    'dilation': quant_conv.dilation if hasattr(quant_conv, 'dilation') else None,
                    'output_padding': quant_conv.output_padding if hasattr(quant_conv, 'output_padding') else None,
                    'groups': quant_conv.groups if hasattr(quant_conv, 'groups') else None,
                }
            else:
                conv_params = None
                print("  quant_conv 不存在")

            if quant_act is not None:
                act_params = {
                    'M_ZERO': quant_act.M_ZERO.cpu().numpy() if hasattr(quant_act, 'M_ZERO') else None,
                    'N_ZERO': quant_act.N_ZERO.cpu().numpy() if hasattr(quant_act, 'N_ZERO') else None,
                    'clip_val': quant_act.clip_val if hasattr(quant_act, 'clip_val') else None,
                }
            else:
                act_params = None
                print("  quant_act 不存在")

            #print(f"Layer {i} quant_conv 权重 shape:", conv_params['weight'].shape if conv_params and conv_params['weight'] is not None else 'None')
            #print(f"Layer {i} quant_act M_ZERO:", act_params['M_ZERO'] if act_params else 'None')

            all_layers_params.append({
                'layer_index': i,
                'quant_conv': conv_params,
                'quant_act': act_params,
            })

    torch.save({
        #'int_model': quantizer.deployment_model.state_dict(),
        'quant_params': all_layers_params,
    }, os.path.join(save_path, 'full_int_model_with_quant_params.pth'))

    print('Deployment model and quant params saved to:', os.path.join(save_path, 'full_int_model_with_quant_params.pth'))


    # if args.quantizer:
    #     print('Exporting INT-only deployment model...')
    #     quantizer.generate_deployment_model()
    #     for i, layer in enumerate(quantizer.param_to_quantize):
    #         act = layer.get('quant_act', None)
    #         if act is not None and hasattr(act, 'M_ZERO') and hasattr(act, 'N_ZERO'):
    #             print(f"Layer {i}:")
    #             print(f"  M0 = {act.M_ZERO}")
    #             print(f"  N0 = {act.N_ZERO}")

    #     #torch.save(quantizer.deployment_model.state_dict(), os.path.join(save_path, 'int_deploy_model.pth'))
    #     #torch.save(quantizer.deployment_model, os.path.join(save_path, 'int_deploy_model_full.pth'))
    #     # torch.save({
    #     #     'int_model': quantizer.deployment_model.state_dict(),
    #     #     'quant_params': quantizer.deployment_model.layer_dict,  # 包含M0, N0等部署用信息
    #     # }, os.path.join(save_path, 'full_int_model_with_quant_params.pth'))

    #     # print('Deployment model saved to:', os.path.join(save_path, 'int_deploy_model.pth'))
    #     torch.save({
    #         'int_model': quantizer.deployment_model.state_dict(),
    #         'quant_params': [
    #             {
    #                 'layer_index': i,
    #                 'M0': layer['quant_act'].M_ZERO,
    #                 'N0': layer['quant_act'].N_ZERO,
    #             } for i, layer in enumerate(quantizer.param_to_quantize)
    #             if 'quant_act' in layer and hasattr(layer['quant_act'], 'M_ZERO') and hasattr(layer['quant_act'], 'N_ZERO')
    #         ],
    #     }, os.path.join(save_path, 'full_int_model_with_quant_params.pth'))






def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None, quantizer=None ):

#    if args.gpus and len(args.gpus) > 1:
#        model = torch.nn.DataParallel(model, args.gpus)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # apply transofrms at the begininng of each epoch

    print('Training: ',training )

    if quantizer is not None and quantizer is not 'deployment':
        quantizer.freeze_BN_and_fold(epoch)

    # input quantization
    n_bits_inpt = 8 #retrieve from quantizer in future version
    max_inpt, min_inpt = 1, -1 #retrieve from quantizer in future version
    n = 2 ** n_bits_inpt - 1
    scale_factor = n / (max_inpt - min_inpt)

    
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            #target = target.cuda(async=True)
            target = target.cuda(non_blocking=True)


        with torch.no_grad():
            input_var = Variable(inputs.type(args.type))
            target_var = Variable(target)

        # quantization before computing output
        if quantizer == 'deployment':
            input_var = input_var.clamp(min_inpt, max_inpt).mul(scale_factor).round()
        elif quantizer is not None:
            input_var = input_var.clamp(min_inpt, max_inpt).mul(scale_factor).round().div(scale_factor)
            quantizer.store_and_quantize(training=training )

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            # restore real value parameters before update
            if quantizer is not None:
                quantizer.backprop_quant_gradients()    
                quantizer.restore_real_value()            

            optimizer.step()
            
        elif quantizer is not None and quantizer is not 'deployment':
            quantizer.restore_real_value()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer, quantizer):
    
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer, quantizer=quantizer )


def validate(data_loader, model, criterion, epoch, quantizer ):
    
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None, quantizer=quantizer)


if __name__ == '__main__':
    main()
