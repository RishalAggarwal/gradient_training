import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import sys
import imp
import molgrid
import argparse
import os
import wandb


def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Model template python file")
    parser.add_argument( '--train_types', type=str, required=True,
                        help="training types file")
    parser.add_argument('--test_types', type=str, required=True,
                        help="test types file")
    parser.add_argument('-recmol', '--recmolcache', type=str, required=True,
                        help="molcache file for recepter")
    parser.add_argument('-ligmol', '--ligmolcache', type=str, required=True,
                        help="molcache file for ligand")
    parser.add_argument('-i', '--iterations', type=int, required=False,
                        help="Number of iterations to run,default 10,000", default=10000)
    parser.add_argument('-b', '--batch_size', type=int, required=False,
                        help="Batch size for training, default 50", default=50)
    parser.add_argument('-s', '--seed', type=int,required=False, help="Random seed, default 0", default=0)
    parser.add_argument('-w', '--weight', type=int,required=False, help="Weight ratio between gradient loss and RMSD loss", default=1)
    parser.add_argument('-t', '--test_interval', type=int, help="How frequently to test (iterations), default 1000",
                        default=1000)
    parser.add_argument('-o', '--outprefix', type=str, help="Prefix for output files",required=True)
    parser.add_argument('--percent_reduced', type=float, default=100,
                        help='Create a reduced set on the fly based on types file, using the given percentage: to use 10 percent pass 10. Range (0,100). Incompatible with --reduced')
    parser.add_argument('--checkpoint', type=str, required=False, help="file to continue training from")
    parser.add_argument('--solver', type=str, help="Solver type. Default is SGD, Nesterov or Adam", default='SGD')
    parser.add_argument('--step_reduce', type=float,
                        help="Reduce the learning rate by this factor with dynamic stepping, default 0.1",
                        default=0.1)
    parser.add_argument('--step_end_cnt', type=float, help='Terminate training after this many lr reductions',
                        default=3)
    parser.add_argument('--step_when', type=int,
                        help="Perform a dynamic step (reduce base_lr) when training has not improved after this many test iterations, default 15",
                        default=15)
    parser.add_argument('--base_lr', type=float, help='Initial learning rate, default 0.01', default=0.01)
    parser.add_argument('--momentum', type=float, help="Momentum parameters, default 0.9", default=0.9)
    parser.add_argument('--weight_decay', type=float, help="Weight decay, default 0.001", default=0.001)
    parser.add_argument('--clip_gradients', type=float, default=10.0, help="Clip gradients threshold (default 10)")
    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)

def initialize_model(model,args):

    def weights_init(m):
        '''initialize model weights with xavier'''
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)
    if args.checkpoint:
        checkpoint=torch.load(args.checkpoint)
        model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.apply(weights_init)


def get_model_gmaker_eproviders(args):
    #train example provider
    eptrain=molgrid.ExampleProvider(shuffle=True, stratify_receptor=True, labelpos=0, stratify_pos=0, stratify_min=0, stratify_max=12, stratify_step=2, recmolcache=args.recmolcache, ligmolcache=args.ligmolcache,data_root='/net/pulsar/home/koes/rishal/rmsd_paper/pdbbind/general_minus_refined')
    eptrain.populate(args.train_types)
    #test example provider
    eptest = molgrid.ExampleProvider(shuffle=True, stratify_receptor=True, labelpos=0, stratify_pos=0, stratify_min=0,
                                      stratify_max=12, stratify_step=2, recmolcache=args.recmolcache,
                                      ligmolcache=args.ligmolcache,data_root='/net/pulsar/home/koes/rishal/rmsd_paper/pdbbind/general_minus_refined')
    eptest.populate(args.test_types)
    #gridmaker with defaults
    gmaker = molgrid.GridMaker()
    dims = gmaker.grid_dimensions(eptrain.num_types())
    model_file = imp.load_source("model", args.model)
    #load model with seed
    torch.manual_seed(args.seed)
    model=model_file.Model(dims)

    return model, gmaker, eptrain, eptest

def train_and_test(args,model,eptrain,eptest,gmaker):

    def test_model(model,ep,gmaker,percent_reduced,batch_size):
        #loss accumulation
        total_loss_mean = []
        rmsd_loss_mean = []
        atomic_grads_loss_mean=[]
        #testing setup
        #testing loop
        for j in range(int((percent_reduced/100) * ep.size())):
            batch = ep.next_batch(batch_size)
            disp_vecs = []
            batch.extract_label(0, float_labels)
            labels = float_labels.to('cuda')
            for b in range(batch_size):
                try:
                    disp_vecs.append(batch[b].coord_sets[0].coords.tonumpy() - batch[b].coord_sets[2].coords.tonumpy())
                    disp_vecs[-1]=torch.from_numpy(disp_vecs[-1]).cuda()
                    batch[b].coord_sets.__delitem__(0)
                except:
                    print(batch[b].coord_sets[0].coords.tonumpy().shape,batch[b].coord_sets[2].coords.tonumpy().shape)
                    disp_vecs.append(torch.zeros(batch[b].coord_sets[2].coords.tonumpy().shape,dtype=torch.float32).cuda())
                    batch[b].coord_sets.__delitem__(0)
                    continue
            #testing
            gmaker.forward(batch, input_tensor,0,random_rotation=False)
            output = model(input_tensor)
            #sending true RMSD values for grid gradientss
            labels=labels.unsqueeze(1)
            #hook=output.register_hook(lambda grad: labels)
            gradspred, = torch.autograd.grad(output, input_tensor,
                                             grad_outputs=output.data.new(output.shape).fill_(1),
                                             create_graph=True)
            gradspred=gradspred.detach()
            atomic_grad_losses = []
            total_losses=[]
            #rmsd losses for the entire batch
            rmsd_losses = (labels - output) ** 2
            rmsd_losses=rmsd_losses.detach().cpu()
            for b in range(batch_size):
                atomic_grads = torch.zeros(disp_vecs[b].shape, dtype=torch.float32, device='cuda')
                if not torch.allclose(disp_vecs[b],atomic_grads):
                    gmaker.backward(batch[b].coord_sets[-1].center(), batch[b].coord_sets[-1], gradspred[b,:14],
                                atomic_grads)
                    pred_grads=F.normalize(atomic_grads,p=1)
                    true_grads=F.normalize(disp_vecs[b],p=1)
                #atomic grad loss per example
                atomic_grads_loss=torch.mean(criteria(true_grads,pred_grads),dim=0).detach().cpu()
                #total_loss for batch
                total_losses.append((rmsd_losses[b] + 10 * atomic_grads_loss))
                #atomic loss for batch
                atomic_grad_losses.append(atomic_grads_loss)
                #mean losses from all batches
                total_loss_mean.append(torch.mean(torch.stack(total_losses)).cpu())
                rmsd_loss_mean.append(torch.mean(rmsd_losses).cpu())
                atomic_grads_loss_mean.append(torch.mean(torch.stack(atomic_grad_losses)).cpu())
        #mean loss for testing session
        total_test_loss_mean=torch.mean(torch.stack(total_loss_mean)).cpu()
        rmsd_test_loss_mean = torch.mean(torch.stack(rmsd_loss_mean)).cpu()
        atomic_test_grads_loss_mean = torch.mean(torch.stack(atomic_grads_loss_mean)).cpu()

        return total_test_loss_mean,rmsd_test_loss_mean,atomic_test_grads_loss_mean




    checkpoint=None
    if args.checkpoint:
        checkpoint=torch.load(args.checkpoint)
    initialize_model(model,args)
    wandb.watch(model)
    iterations = args.iterations
    test_interval = args.test_interval
    batch_size=args.batch_size
    percent_reduced= args.percent_reduced
    outprefix=args.outprefix
    prev_total_loss_snap=''
    prev_rmsd_loss_snap=''
    prev_grad_loss_snap=''
    prev_snap=''
    initial=0
    if args.checkpoint:
        initial=checkpoint['Iteration']
    last_test=0

    if 'SGD' in args.solver:
        optimizer=torch.optim.SGD(model.parameters(),lr=args.base_lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif 'Nesterov' in args.solver:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay,nesterov=True)
    elif 'Adam' in args.solver:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.base_lr,weight_decay=args.weight_decay)
    else:
        print("No valid solver argument passed (SGD, Adam, Nesterov)")
        sys.exit(1)
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=args.step_reduce,patience=args.step_when,verbose=True)
    if args.checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    Bests={}
    Bests['train_iteration']=0
    Bests['test_grad_loss']=torch.from_numpy(np.asarray(np.inf))
    Bests['test_rmsd_loss']=torch.from_numpy(np.asarray(np.inf))
    Bests['test_total_loss']=torch.from_numpy(np.asarray(np.inf))
    if args.checkpoint:
        Bests=checkpoint['Bests']

    dims = gmaker.grid_dimensions(eptrain.num_types())
    tensor_shape = (batch_size,) + dims

    model.cuda()
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda',requires_grad=True).contiguous()
    float_labels = torch.zeros(batch_size, dtype=torch.float32,device='cuda').contiguous()
    gradspred_grad = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda',requires_grad=False).contiguous()
    criteria=torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')
    cos = nn.CosineSimilarity(dim=1, eps=1e-10)

    for i in range(initial,iterations):
        batch = eptrain.next_batch(batch_size)
        transformers=[]
        disp_vecs=[]
        batch.extract_label(0, float_labels)
        labels = float_labels.to('cuda')
        for b in range(batch_size):
            #faulty examples (don't know why)
            try:
                disp_vecs.append(batch[b].coord_sets[0].coords.tonumpy() - batch[b].coord_sets[2].coords.tonumpy())
                disp_vecs[-1]=torch.from_numpy(disp_vecs[-1]).cuda()
                batch[b].coord_sets.__delitem__(0)
            except:
                print(batch[b].coord_sets[0].coords.tonumpy().shape,batch[b].coord_sets[2].coords.tonumpy().shape)
                disp_vecs.append(torch.zeros(batch[b].coord_sets[2].coords.tonumpy().shape,dtype=torch.float32).cuda())
                batch[b].coord_sets.__delitem__(0)
            transformer=molgrid.Transform(batch[b].coord_sets[-1].center(),6,True)
            #doesnt change underlying coordinate
            gmaker.forward(batch[b],transformer,input_tensor[b])
            transformers.append(transformer)

        labels = labels.reshape(batch_size,1)
        labels=labels.contiguous()
        optimizer.zero_grad()
        output = model(input_tensor)
        #sending true RMSD values for grid gradientss
        #hook=output.register_hook(lambda grad: labels)
        gradspred, = torch.autograd.grad(output, input_tensor,
                                   grad_outputs=output.data.new(output.shape).fill_(1),
                                   create_graph=True)
        #losses for the batch
        atomic_grad_losses = []
        #rmsd_losses = (labels-output) ** 2
        labels=labels.contiguous()
        rmsd_losses=criteria(output,labels)
        total_losses=[]
        for b in range(batch_size):
            atomic_grads=torch.zeros(disp_vecs[b].shape, dtype=torch.float32, device='cuda',requires_grad=True)
            atomic_grads1=torch.zeros(disp_vecs[b].shape, dtype=torch.float32, device='cuda')
            #apply transform to underlying coords
            transformers[b].forward(batch[b],batch[b])
            if not torch.allclose(disp_vecs[b],atomic_grads):
                gmaker.backward(transformers[b].get_rotation_center(),batch[b].coord_sets[-1],gradspred[b,:14],atomic_grads)
                pred_grads=F.normalize(atomic_grads,p=1)
                true_grads=F.normalize(disp_vecs[b],p=1)
                cost=criteria(pred_grads,true_grads)
                #cost=1 - torch.cosine_similarity(atomic_grads,disp_vecs[b],dim=1)
                cost.mean().backward()
                batch[b].coord_sets[1].make_vector_types()
                type_grad = torch.zeros(batch[b].coord_sets[1].type_vector.tonumpy().shape,dtype=torch.float32,device='cuda')
                gmaker.backward_gradients(transformers[b].get_rotation_center(),batch[b].coord_sets[1],gradspred[b,:14],atomic_grads.grad,type_grad,gradspred_grad[b,:14],atomic_grads1,type_grad)
            #atomic loss for example
            atomic_grads_loss=cost.mean().cpu()
            #atomic losses for batch
            atomic_grad_losses.append(atomic_grads_loss)
            #total losses for batch
            total_losses.append(rmsd_losses[b]+ 10*atomic_grads_loss)

        #total loss for batch
        total_loss_mean = torch.mean(torch.stack(total_losses).cpu())
        #rmsd loss for batch
        rmsd_loss_mean = torch.mean(rmsd_losses)
        #atomic loss for batch
        atomic_grads_loss_mean = torch.mean(torch.stack(atomic_grad_losses).cpu())
        '''gradspred*=gradspred_grad
        gradspred=gradspred.contiguous()
        loss = rmsd_losses.mean() + gradspred.contiguous().sum()
        loss = loss.contiguous()
        input_tensor=input_tensor.contiguous()
        loss.backward()'''
        gradspred_grad = gradspred_grad * args.weight
        gradspred.backward(gradspred_grad,retain_graph=True)
        #hook.remove()
        rmsd_losses.mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
        optimizer.step()
        
       
        wandb.log({'total_train_loss': torch.sqrt(total_loss_mean), 'iteration': i+1, 'rmsd_train_loss': torch.sqrt(rmsd_loss_mean),'atomic_grad_train_loss': atomic_grads_loss_mean})
        
        if i%test_interval==0 and i!=0 :

            total_loss_mean, rmsd_loss_mean, atomic_grads_loss_mean = test_model(model, eptest, gmaker,
                                                                                 percent_reduced,batch_size)
            scheduler.step(total_loss_mean)
            print('done')
            if total_loss_mean<Bests['test_total_loss']:
                Bests['test_total_loss']=total_loss_mean
                wandb.run.summary["total_test_test_loss"]=torch.sqrt(Bests['test_total_loss'])
                Bests['train_iteration']=i
                if Bests['train_iteration']-i>=args.step_when and optimizer.param_groups[0]['lr']<= ((args.step_reduce)**args.step_end_cnt)*args.base_lr:
                    last_test=1
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i+1}, outprefix+'_best_total_'+str(i+1)+'.pth.tar')
                if prev_total_loss_snap:
                    os.remove(prev_total_loss_snap)
                prev_total_loss_snap=outprefix+'_best_total_'+str(i+1)+'.pth.tar'
            if rmsd_loss_mean<Bests['test_rmsd_loss']:
                Bests['test_rmsd_loss']=rmsd_loss_mean
                wandb.run.summary["rmsd_test_test_loss"]=torch.sqrt(Bests['test_rmsd_loss'])
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i + 1},outprefix+'_best_rmsd_'+str(i+1)+'.pth.tar')
                if prev_rmsd_loss_snap:
                    os.remove(prev_rmsd_loss_snap)
                prev_rmsd_loss_snap = outprefix + '_best_rmsd_' + str(i + 1) + '.pth.tar'
            if atomic_grads_loss_mean<Bests['test_grad_loss']:
                Bests['test_grad_loss']=atomic_grads_loss_mean
                wandb.run.summary["atomic_grad_test_test_loss"]=Bests['test_grad_loss']
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i + 1},outprefix+'_best_atom_'+str(i+1)+'.pth.tar')
                if prev_grad_loss_snap:
                    os.remove(prev_grad_loss_snap)
                prev_grad_loss_snap = outprefix + '_best_atom_' + str(i + 1) + '.pth.tar'
            print(
                "Iteration {}, total_test_loss: {:.3f},rmsd_test_loss: {:.3f},grad_test_loss: {:.3f},  Best_total_loss: {:.3f},Best_rmsd_loss: {:.3f},Best_grad_loss: {:.3f},learning_Rate: {:.7f}".format(
                    i + 1, torch.sqrt(total_loss_mean),torch.sqrt(rmsd_loss_mean),torch.sqrt(atomic_grads_loss_mean), torch.sqrt(Bests['test_total_loss']),torch.sqrt(Bests['test_rmsd_loss']),torch.sqrt(Bests['test_grad_loss']),optimizer.param_groups[0]['lr']))
            wandb.log({'total_test_test_loss': torch.sqrt(total_loss_mean), 'iteration': i + 1,'rmsd_test_test_loss': torch.sqrt(rmsd_loss_mean),'atomic_grad_test_test_loss': atomic_grads_loss_mean,'learning rate':optimizer.param_groups[0]['lr']})
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Iteration': i + 1}, outprefix + '_' + str(i + 1) + '.pth.tar')
            if prev_snap:
                os.remove(prev_snap)
            prev_snap = outprefix + '_' + str(i + 1) + '.pth.tar'
        if last_test:
            return Bests


if __name__ == '__main__':
    wandb.init(project="gradient_training")
    (args,cmdline) = parse_args()
    model, gmaker, eptrain, eptest = get_model_gmaker_eproviders(args)
    Bests=train_and_test(args,model,eptrain,eptest,gmaker)
    print(Bests)







