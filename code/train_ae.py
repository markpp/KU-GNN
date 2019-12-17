import os
import numpy as np
import torch
import torch.optim as optim
from cvae_pointnet import *
from optimizer import Lookahead, RAdam

np.random.seed(42)
torch.manual_seed(42)

batch_size = 64

def predict(X, model, dev, crit):
    model.eval() # Switch to evaluation mode
    losses = []
    zs = []
    recs = []
    for x in X:
        x = x.unsqueeze(0)
        data = x.to(dev).transpose(2, 1)

        # Run data through model and calc loss
        rec, mu, logvar, z = model(data)
        loss = crit(rec, data, mu, logvar)

        if dev.type == 'cuda':
            rec = rec.cpu()
            z = z.cpu()
        rec = rec.transpose(2, 1)[0]
        z = z[0]

        losses.append(loss.item())
        zs.append(z.data.numpy())
        recs.append(rec.data.numpy())
    return np.array(losses), np.array(zs), np.array(recs)


def train(X, model, opt, dev, crit):
    model.train() # Switch to training mode
    total_loss = 0
    num_batches = 0

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        data = X[indices].to(dev).transpose(2, 1)


        # Run data through model and calc loss
        rec, mu, logvar, z = model(data)
        loss = crit(rec, data, mu, logvar)

        # Optimize using the loss
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        num_batches += 1
        epoch_loss = total_loss / num_batches
    return epoch_loss

def evaluate(X, model, dev, crit):
    model.eval() # Switch to evaluation mode
    total_loss = 0
    num_batches = 0

    permutation = np.random.permutation(X.shape[0])
    for i in range(0,X.shape[0], batch_size):
        indices = permutation[i:i+batch_size]
        data = X[indices].to(dev).transpose(2, 1)

        # Run data through model and calc loss
        rec, mu, logvar, z = model(data)
        loss = crit(rec, data, mu, logvar)


        total_loss += loss.item()
        num_batches += 1
        epoch_loss = total_loss / num_batches

    return epoch_loss


#Epoch #155, validation err: 0.03922 (best: 0.03922), train err: 0.03711
#Epoch #150, validation err: 0.03921 (best: 0.03921), train err: 0.03710
#Epoch #2290, validation err: 0.03919 (best: 0.03919), train err: 0.03694

if __name__ == '__main__':
    TRAIN = 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir = "output/ae"
    # load data #
    data_source = "kin"
    val_X = np.load('{}/{}/{}_x_1024.npy'.format("input",data_source,"val"),allow_pickle=True)
    train_X = np.load('{}/{}/{}_x_1024.npy'.format("input",data_source,"train"),allow_pickle=True)


    train_x, val_x = torch.from_numpy(train_X[:,:,:3]), torch.from_numpy(val_X[:,:,:3])

    crit = loss()

    if TRAIN:
        model = cvae().to(dev)

        #opt = optim.Adam(model.parameters(), lr=10e-5, weight_decay=5e-4)
        opt = Lookahead(base_optimizer=RAdam(model.parameters(), lr=10e-4),k=5,alpha=0.5)

        plot_file = open("{}/err.txt".format(dir),'w')
        plot_file.write("train_err:val_err\n")

        best_val_loss = 999.9

        for epoch in range(301):
            train_loss = train(train_x, model, opt, dev, crit)
            print('Epoch #{}, training loss {:.5f}'.format(epoch,train_loss))

            if epoch % 5 == 0:
                with torch.no_grad():
                    val_loss = evaluate(val_x, model, dev, crit)
                    plot_file.write("{:.5f}:{:.5f}\n".format(train_loss, val_loss))

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        #torch.save(model.state_dict(),'{}/model.pth'.format(dir))
                        torch.save(model,'{}/model.pkl'.format(dir))
                        print('Epoch #{}, validation err: {:.5f} (best: {:.5f}), train err: {:.5f}'.format(epoch, val_loss, best_val_loss, train_loss))
        plot_file.close()
    else:
        model = torch.load("{}/model.pkl".format(dir))
        losses, zs, recs = predict(val_x, model, dev, crit)
        print(losses.shape)
        print(zs.shape)
        print(recs.shape)
        np.save("{}/val_losses.npy".format(dir),losses)
        np.save("{}/val_zs.npy".format(dir),zs)
        np.save("{}/val_recs.npy".format(dir),recs)

        '''
        losses, zs, recs = predict(train_x, model, dev, crit)
        print(losses.shape)
        print(zs.shape)
        print(recs.shape)
        np.save("{}/train_losses.npy".format(dir),losses)
        np.save("{}/train_zs.npy".format(dir),zs)
        np.save("{}/train_recs.npy".format(dir),recs)
        '''
