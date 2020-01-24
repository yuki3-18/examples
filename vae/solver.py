from topologylayer.nn import *
from utils import init_tri_complex_3d
from torch.nn import functional as F
import torch
import numpy as np

def loss_function(recon_x, x, mu, logvar):
    batch_size = x.size(0)
    feature_size = x.size(1)
    assert batch_size != 0

    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 729), reduction='sum')
    MSE = F.mse_loss(recon_x, x, size_average=False).div(batch_size)*feature_size
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    if args.topo==True:
        topo, b01, b0, b1, b2 = topological_loss(recon_x, x)
        return MSE + KLD*args.beta + topo*args.ramda
    else:
        return MSE + KLD*args.beta

def topological_loss(recon_x, x):
    batch_size = x.size(0)
    b01 = 0
    b0 = 0
    b1 = 0
    b2 = 0
    cpx = init_tri_complex_3d(9, 9, 9)
    layer = LevelSetLayer(cpx, maxdim=2, sublevel=False)
    f01 = TopKBarcodeLengths(dim=0, k=1)
    f0 = PartialSumBarcodeLengths(dim=0, skip=1)
    f1 = SumBarcodeLengths(dim=1)
    f2 = SumBarcodeLengths(dim=2)
    for i in range(batch_size):
        dgminfo = layer(recon_x.view(batch_size, 9, 9, 9)[i])
        b01 += ((1 - f01(dgminfo)) ** 2).sum()
        b0 += (f0(dgminfo) ** 2).sum()
        b1 += (f1(dgminfo) ** 2).sum()
        b2 += (f2(dgminfo) ** 2).sum()
    b01 = b01.div(batch_size)
    b0 = b0.div(batch_size)
    b1 = b1.div(batch_size)
    b2 = b2.div(batch_size)
    topo = b01 + b0 + b1 + b2
    return topo, b01, b0, b1, b2

def train(epoch):
    model.train()
    train_loss = 0
    b01 = 0
    b0 = 0
    b1 = 0
    b2 = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        # loss = loss_function(recon_batch, data, mu, logvar)
        loss,l01,l0,l1,l2 = topological_loss(recon_batch, data)
        loss.backward()
        train_loss += loss.item()
        b01 += l01.item()
        b0 += l0.item()
        b1 += l1.item()
        b2 += l2.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    train_loss /= len(train_loader.dataset)
    train_loss_list.append(train_loss)
    b01 /= len(train_loader.dataset)
    b0 /= len(train_loader.dataset)
    b1 /= len(train_loader.dataset)
    b2 /= len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss))

    writer.add_scalars("loss/topological_loss", {'topo': train_loss,
                                                 'b01': b01,
                                                 'b0': b0,
                                                 'b1': b1,
                                                 'b2': b2}, epoch)
    viz.line(X=np.array([epoch]), Y=np.array([train_loss]), win='topo_loss', name='topo', update='append',
             opts=dict(showlegend=True))
    viz.line(X=np.array([epoch]), Y=np.array([b01]), win='topo_loss', name='b01', update='append')
    viz.line(X=np.array([epoch]), Y=np.array([b0]), win='topo_loss', name='b0', update='append')
    viz.line(X=np.array([epoch]), Y=np.array([b1]), win='topo_loss', name='b1', update='append')
    viz.line(X=np.array([epoch]), Y=np.array([b2]), win='topo_loss', name='b2', update='append')

    return train_loss


def val(epoch):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss, _, _, _, _ = topological_loss(recon_batch, data)
            val_loss = loss.item()
            # val_loss += loss_function(recon_batch, data, mu, logvar).item()
            # if i == 0 and epoch>200:
            #     n = min(data.size(0), 11)
            #     comparison = torch.cat([data[:n],
            #                           recon_batch.view(args.batch_size, 729)[:n]])
            #     ori = np.reshape(data[:n].cpu().numpy(), [n, 9, 9, 9])
            #     rec = np.reshape(recon_batch[:n].cpu().numpy(), [n, 9, 9, 9])
            #     visualize_slices(ori[:,4,:], rec[:,4,:], outdir + 'reconstruction_' + str(epoch))
                # save_image(comparison.cpu(),
                #          'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    val_loss /= len(val_loader.dataset)
    val_loss_list.append(val_loss)
    print('====> val set loss: {:.4f}'.format(val_loss))

    return val_loss