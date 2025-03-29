import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *

tqdm.monitor_interval = 0

class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()

        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        return self.state.get(name)

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss'].value()[0]
        if display:
            if training:
                print(f'Epoch: [{self.state["epoch"]}]\tLoss {loss:.4f}')
            else:
                print(f'Test: \t Loss {loss:.4f}')
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['loss_batch'] = self.state['loss'].item()
        self.state['meter_loss'].add(self.state['loss_batch'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print(f'Epoch: [{self.state["epoch"]}][{self.state["iteration"]}/{len(data_loader)}]\t'
                      f'Time {self.state["batch_time_current"]:.3f} ({batch_time:.3f})\t'
                      f'Data {self.state["data_time_batch"]:.3f} ({data_time:.3f})\t'
                      f'Loss {self.state["loss_batch"]:.4f} ({loss:.4f})')
            else:
                print(f'Test: [{self.state["iteration"]}/{len(data_loader)}]\t'
                      f'Time {self.state["batch_time_current"]:.3f} ({batch_time:.3f})\t'
                      f'Data {self.state["data_time_batch"]:.3f} ({data_time:.3f})\t'
                      f'Loss {self.state["loss_batch"]:.4f} ({loss:.4f})')

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            with torch.no_grad():
                self.state['output'] = model(input_var)
                self.state['loss'] = criterion(self.state['output'], target_var)
        else:
            self.state['output'] = model(input_var)
            self.state['loss'] = criterion(self.state['output'], target_var)
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):
        if self._state('train_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['train_transform'] = transforms.Compose([
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        if self._state('val_transform') is None:
            normalize = transforms.Normalize(mean=model.image_normalization_mean,
                                             std=model.image_normalization_std)
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):
        self.init_learning(model, criterion)

        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # Optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print(f"=> loading checkpoint '{self.state['resume']}'")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                checkpoint = torch.load(self.state['resume'], map_location=device)
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print(f"=> loaded checkpoint '{self.state['resume']}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{self.state['resume']}'")

        if self.state['use_gpu'] and torch.cuda.is_available():
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True
            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            lr = self.adjust_learning_rate(optimizer)
            print('lr:', lr)

            self.train(train_loader, model, criterion, optimizer, epoch)
            prec1 = self.validate(val_loader, model, criterion)

            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] and torch.cuda.is_available() else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(f' *** best={self.state["best_score"]:.3f}')
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):
        model.train()
        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu'] and torch.cuda.is_available():
                self.state['target'] = self.state['target'].cuda(non_blocking=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):
        model.eval()
        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu'] and torch.cuda.is_available():
                self.state['target'] = self.state['target'].cuda(non_blocking=True)

            self.on_forward(False, model, criterion, data_loader)

            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            self.on_end_batch(False, model, criterion, data_loader)

        return self.on_end_epoch(False, model, criterion, data_loader)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename = os.path.join(self.state['save_model_path'], filename)
            os.makedirs(self.state['save_model_path'], exist_ok=True)
        print(f'save model {filename}')
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') and self._state('filename_previous_best'):
                os.remove(self._state('filename_previous_best'))
            filename_best = os.path.join(self.state['save_model_path'] or '', f'model_best_{state["best_score"]:.4f}.pth.tar')
            shutil.copyfile(filename, filename_best)
            self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_list = []
        decay = 0.1 if sum(self.state['epoch'] == np.array(self.state['epoch_step'])) > 0 else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay
            lr_list.append(param_group['lr'])
        return np.unique(lr_list)

class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                print(f'Epoch: [{self.state["epoch"]}]\tLoss {loss:.4f}\tmAP {map:.3f}')
                print(f'OP: {OP:.4f}\tOR: {OR:.4f}\tOF1: {OF1:.4f}\tCP: {CP:.4f}\tCR: {CR:.4f}\tCF1: {CF1:.4f}')
            else:
                print(f'Test: \t Loss {loss:.4f}\t mAP {map:.3f}')
                print(f'OP: {OP:.4f}\tOR: {OR:.4f}\tOF1: {OF1:.4f}\tCP: {CP:.4f}\tCR: {CR:.4f}\tCF1: {CF1:.4f}')
                print(f'OP_3: {OP_k:.4f}\tOR_k: {OR_k:.4f}\tOF1_3: {OF1_k:.4f}\tCP_3: {CP_k:.4f}\tCR_3: {CR_k:.4f}\tCF1_3: {CF1_k:.4f}')
        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print(f'Epoch: [{self.state["epoch"]}][{self.state["iteration"]}/{len(data_loader)}]\t'
                      f'Time {self.state["batch_time_current"]:.3f} ({batch_time:.3f})\t'
                      f'Data {self.state["data_time_batch"]:.3f} ({data_time:.3f})\t'
                      f'Loss {self.state["loss_batch"]:.4f} ({loss:.4f})')
            else:
                print(f'Test: [{self.state["iteration"]}/{len(data_loader)}]\t'
                      f'Time {self.state["batch_time_current"]:.3f} ({batch_time:.3f})\t'
                      f'Data {self.state["data_time_batch"]:.3f} ({data_time:.3f})\t'
                      f'Loss {self.state["loss_batch"]:.4f} ({loss:.4f})')

class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()

        if not training:
            with torch.no_grad():
                self.state['output'] = model(feature_var, inp_var)
                self.state['loss'] = criterion(self.state['output'], target_var)
        else:
            self.state['output'] = model(feature_var, inp_var)
            self.state['loss'] = criterion(self.state['output'], target_var)
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]