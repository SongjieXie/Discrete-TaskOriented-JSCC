from datetime import datetime
from typing import Iterable
import torch 
from tensorboardX import SummaryWriter
from utils.accuracy import accuracy

def train_one_epoch(dataloader:Iterable,  model:torch.nn.Module,
                    optimizer:torch.optim.Optimizer, criterion:torch.nn.Module,
                    writer: SummaryWriter, epoch: int, mod, args):
    model.train()
    batches_start = datetime.now()
    for i_batch, (imgs, labs) in enumerate(dataloader):
        imgs = imgs.to(args.device)
        labs = labs.to(args.device)
        outs, dist = model(imgs, mod=mod)
        if True:
            loss, loss_post, loss_entr = criterion(dist, outs, labs)
        optimizer.zero_grad()
        loss.backward()
        if args.maxnorm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.maxnorm)
        optimizer.step()

        acc = accuracy(outs, labs, (1,3))
        
        writer.add_scalar('train/losses', loss.item(), args.n_iter)
        writer.add_scalar('train/losses_post', loss_post.item(), args.n_iter)
        writer.add_scalar('train/losses_entr', loss_entr.item(), args.n_iter)
        writer.add_scalar('train/acc1',acc[0].item(), args.n_iter)
        writer.add_scalar('train/acc3',acc[1].item(), args.n_iter)
        args.n_iter += 1
        
        if i_batch %80 == 0:
            batches_end = datetime.now()
            avg_time = (batches_end - batches_start)/80
            print('\n \n average batch time for batch size of', imgs.shape[0],':', avg_time)
            batches_start = datetime.now()
            print('[%d][%d/%d]\t Losses:%.4f\t Loss_post: %.4f\t Loss_entr: %.4f\t'
                  %(epoch, i_batch, len(dataloader), loss.item(), loss_post.item(), loss_entr.item()))
            print('acc1:%.4f\t acc3:%.4f'%(acc[0].item(), acc[1].item()))
        
def test(dataloader, model, criterion, writer, epoch, mod, args):
    losses, losses_post, losses_entr, acc1, acc3 = 0., 0., 0., 0., 0.
    with torch.no_grad():
        model.eval()
        print("testing")
        for imgs, labs in dataloader:
            imgs = imgs.to(args.device)
            labs = labs.to(args.device)
            
            outs, dist = model(imgs, mod=mod)
            if True:
                loss, loss_post, loss_entr = criterion(dist, outs, labs)
                acc = accuracy(outs, labs, (1,3))
                acc1 += acc[0].item()
                acc3 += acc[1].item()
                writer.add_scalar('test/acc1-single', acc[0].item(), epoch)
                writer.add_scalar('test/acc3-single', acc[1].item(), epoch)
            
            losses += loss.item()
            losses_post += loss_post.item()
            losses_entr += loss_entr.item()
            
    writer.add_scalar('test/losses', losses/len(dataloader), epoch)
    writer.add_scalar('test/losses_post', losses_post/len(dataloader), epoch)
    writer.add_scalar('test/losses_entr', losses_entr/len(dataloader), epoch)
    writer.add_scalar('test/acc1', acc1/len(dataloader), epoch)
    writer.add_scalar('test/acc3', acc3/len(dataloader), epoch)
    
    print('Done!')
    
    return acc1/len(dataloader)
            
if __name__ == '__main__ ':
    print("here")          
