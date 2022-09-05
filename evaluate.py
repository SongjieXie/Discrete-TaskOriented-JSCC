import os
import torch
from datasets.dataloader import get_data
from model.DT_JSCC import DTJSCC_CIFAR10, DTJSCC_MNIST
from utils.modulation import QAM, PSK
from utils.accuracy import accuracy

def eval_test(datal, model, mod, args):
    acc1, acc3 = 0., 0.
    with torch.no_grad():
        model.eval()
        for imgs, labs in datal:
            imgs = imgs.to(args.device)
            labs = labs.to(args.device)
            
            outs, dist = model(imgs, mod=mod)
            acc = accuracy(outs, labs, (1,3))
            acc1 += acc[0].item()
            acc3 += acc[1].item()
    print('Done!')
    
    return acc1/len(datal), acc3/len(datal), dist

def main(args):
    # model and dataloader
    if args.dataset == 'MNIST':
        model = DTJSCC_MNIST(args.latent_d, 10, args.num_latent, args.num_embeddings)
    elif args.dataset == 'CIFAR10':
        model = DTJSCC_CIFAR10(3, 512, 10, num_embeddings=args.num_embeddings)
    else:
        assert 1 == 0, args.dataset


    dataloader_vali = get_data(args.dataset, 256, n_worker=8, train=False)
    # load model
    checkpoint = torch.load(path_to_backup, map_location='cpu')
    model.load_state_dict(checkpoint['model_states'])
    model.to(args.device)
    
    PSNRs = list(range(2, 21, 2))
    acc1s = []
    acc3s = []
    dist_re = None
    for psnr in PSNRs:
        if args.mod == 'qam':
            mod = QAM(args.num_embeddings, psnr)
        elif args.mod == 'psk':
            mod = PSK(args.num_embeddings, psnr)
        a1, a3, dist = eval_test(dataloader_vali, model, mod, args)
        acc1s.append(a1)
        acc3s.append(a3)
        dist_re = dist
    with open('{0}.pt'.format(path_to_save), 'wb') as f:
        torch.save(
                {
                    'acc1s':acc1s,
                    'acc3s':acc3s,
                    'dist':dist_re
                }, f
            )
        
if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description='ways')
    parser.add_argument('--mod', type=str, default='psk', help='The modulation')
    parser.add_argument('--num_latent', type=int, default=4, help='The number of latent variable')
    parser.add_argument('--latent_d', type=int, default=512, help='The dimension of latent vector')

    parser.add_argument('-d', '--dataset', type=str, default='CIFAR10', help='dataset name')
    parser.add_argument('-r', '--root', type=str, default='./trainded_models', help='The root of trained models')
    parser.add_argument('--device', type=str, default='cuda:0', help= 'The device')
    parser.add_argument('--num_embeddings', type=int, default=16, help='The size of codebook')

    parser.add_argument('--name', type=str, default='ta', help= 'The trained model')
    parser.add_argument('--save_root', type=str, default='./results-2', help= 'The root of result')
    
    args = parser.parse_args()
    path_to_backup = os.path.join(args.root, args.name)
    path_to_backup = '{0}/best.pt'.format(path_to_backup)
    
    if not os.path.exists(args.save_root):
        print('making results...')
        os.makedirs(args.save_root)
        
    path_to_save = os.path.join(args.save_root, args.name)
        
    device = torch.device(args.device if(torch.cuda.is_available()) else "cpu")
    print('Device: ', device)
    
    main(args)
    
    
        

    
        
    
    
    

