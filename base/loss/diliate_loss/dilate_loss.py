import torch

from base.loss.diliate_loss import soft_dtw, path_soft_dtw


def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk     
	loss_shape = softdtw_batch(D,gamma)
	
	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)           
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output) 
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal
if __name__ == "__main__":
    x = torch.randn(128,10).cuda()
    y = torch.randn(128,10).cuda()
    # x = x.t()
    # y = y.t()
    x = torch.unsqueeze(x,2)
    y = torch.unsqueeze(y,2)
    #x = torch.squeeze(x)
    print(x.shape)
    loss_1 = dilate_loss(outputs=x,targets=y,alpha=0.5,gamma=0.001,device='cpu')
    # loss = sdtw(x, y)  # Just like any torch.nn.xyzLoss()
    print(loss_1[0])
    x = torch.squeeze(x)
    y = torch.squeeze(y)
    import  torch.nn as nn
    loss_mse = nn.MSELoss()
    print(loss_mse(x,y))
    # Aggregate and call backward()
    #loss.mean().backward()