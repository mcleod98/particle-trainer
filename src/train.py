import torch
import time
import gc

from torch.utils.data import DataLoader
from utils import pad_collate, resume_checkpoint, save_checkpoint
from loss import CustomLoss

def train_net(model, trainset, valset, hyper, config):

    trainloader = DataLoader(trainset, batch_size=hyper['batch_size'], shuffle=False, num_workers=0, collate_fn=pad_collate)
    valloader = DataLoader(valset, batch_size=hyper['batch_size'], shuffle=False, num_workers=0, collate_fn=pad_collate)
    loss_cls = CustomLoss(loss_weights=hyper['loss_weights'])
    #loss_cls = torch.nn.MSELoss()

    # Initialize weights
    for layer in model.children():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), hyper['learning_rate'])
    epoch = hyper['epochs']
    start = 0
    
    if len(config['checkpoint_load']) > 0:
        print(f'Resuming from checkpoint at {config["checkpoint_load"]}')
        model, optimizer, start = resume_checkpoint(config['checkpoint_load'], model, optimizer)
    
    
    lastgc = 0
    for i in range(start, epoch):
        print(f'------ Beginning epoch {i} ---------')
        t0 = time.time()
        trainloss = 0
        sumlossx = 0
        sumlossy = 0
        sumlossa = 0
        sumlossb = 0
        sumlossprob = 0
        model.train()
        for batch, x in enumerate(trainloader):
            optimizer.zero_grad()

            img = x['img']
            targets = x['particles']
            
            preds = model(img)
            
            losses = loss_cls(preds, targets)
            lossx= losses['lossx']
            lossy= losses['lossy']
            lossa= losses['lossa']
            lossb= losses['lossb']
            lossprob= losses['lossprob']
            loss = lossx + lossy + lossa + lossb + lossprob

            trainloss += loss.item()
            sumlossx = sumlossx + lossx.item()
            sumlossy = sumlossy + lossy.item()
            sumlossa = sumlossa + lossa.item()
            sumlossb = sumlossb + lossb.item()
            sumlossprob = sumlossprob + lossprob.item()
            
            loss.backward()
            optimizer.step()

            if lastgc > 50:
                lastgc = 0
                gc.collect()
            else:
                lastgc += 1
        #print(f'Epoch {i} completed in {int(time.time() - t0)}s')
        print(f'Epoch {i} completed in {int(time.time() - t0)}s with average training loss: {trainloss/len(trainloader)}')
        print(f'   -- Normalized Losses: Probability: {sumlossprob/trainloss}, x: {sumlossx/trainloss}, y: {sumlossy/trainloss}, a: {sumlossa/trainloss}, b: {sumlossb/trainloss}:')

        valloss = 0
        model.eval()
        for valbatch, val in enumerate(valloader):
            img = val['img']
            targets = val['particles']
            preds = model(img)
            val_losses = loss_cls(preds, targets)
            vlossprob = val_losses['lossprob']
            vlossx= val_losses['lossx']
            vlossy= val_losses['lossy']
            vlossa= val_losses['lossa']
            vlossb= val_losses['lossb']
            L = vlossx + vlossy + vlossa + vlossb + vlossprob
            valloss = valloss + L.item()

            if lastgc > 100:
                lastgc = 0
                gc.collect()
            else:
                lastgc += 1
        print(f'   -- Average validation loss: {valloss/len(valloader)}; val/train loss: {valloss/len(valloader)/(trainloss/len(trainloader))}')
        if len(config['checkpoint_save']) > 0:
            save_checkpoint(config['checkpoint_save'], model, optimizer, i)

    print('Model training complete:')
    if len(config['model_save']) > 0:
        print(f'Saving final model weights to {config["model_save"]}')
        torch.save(model.state_dict(), config['model_save'])