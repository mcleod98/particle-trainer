import torch

class CustomLoss(torch.nn.Module):
    '''
    Custom loss function. 
    
    Use binary cross entropy to calculate loss for probability of particle identification, and mean square error for losses of continuous predicted variables

    Match highest probability predictions to the closest available ground truth targets

    False positive/negatives are penalized by BC
    '''
    def __init__(self, loss_weights):
        super(CustomLoss, self).__init__()
        self.MSE = torch.nn.MSELoss()
        self.BCE = torch.nn.BCELoss()
        self.loss_weights = loss_weights
        
    def forward(self, preds, targets):
        #batch dimension
        b = preds.size(0)
        real_targs = torch.zeros(preds.size(), requires_grad=False)
        for i in range(b):
            built = self.build_targets(preds[i,:], targets[i,:])
            real_targs[i,:] = built

        lossprob = self.BCE(preds[:,:,0], real_targs[:,:,0]) * self.loss_weights[0]
        lossx = self.MSE(preds[:,:,1], real_targs[:,:,1]) * self.loss_weights[1]
        lossy = self.MSE(preds[:,:,2], real_targs[:,:,2]) * self.loss_weights[2]
        lossa = self.MSE(preds[:,:,3], real_targs[:,:,3]) * self.loss_weights[3]
        lossb = self.MSE(preds[:,:,4], real_targs[:,:,4]) * self.loss_weights[4]

        del real_targs
        #return lossx + lossy + lossa + lossb + lossprob
        return {'lossx': lossx, 'lossy': lossy, 'lossa': lossa, 'lossb': lossb, 'lossprob': lossprob}
        
         
    def build_targets(self, preds, targets):
        sort_conf = torch.argsort(preds[:,0], descending=True)
        built_targs = torch.zeros(preds.size(), requires_grad=False, dtype=torch.float)
        nonzero = (targets[:, 0] == 1)
        targets = targets[nonzero]
        n_pos = (preds[:, 0] > .5).nonzero().size(0)
        n_targs = targets.size(0)

        for i, p in enumerate(preds[sort_conf]):
            prob, x, y, a, b = p
            if i >= n_targs:
                built_targs[i,:] = torch.tensor([0, x, y, a, b], requires_grad=False)
            elif i >= n_pos:
                notchosen = torch.lt(targets[:,1], 5000)
                built_targs[i,:] = targets[notchosen][0]
            else:
                #Calculate distance between predicted center and each target's center
                dists = torch.sqrt(torch.square(targets[:,1] - x)  + torch.square(targets[:, 2] - y))

                #Find closest target and copy its values into built_targs
                min_index = torch.argmin(dists)
                built_targs[i, :] = targets[min_index, :]

                #Replace values in target tensor so these values won't get selected again
                targets[min_index,(1, 2)] = 5000
                
        # Have to revert order on _t since it was prepared with sorted indices
        built_targs = built_targs[torch.argsort(sort_conf)]
        return built_targs
