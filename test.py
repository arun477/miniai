from datasets import load_dataset
import torchvision.transforms.functional as TF
import torch
from torch import nn,tensor
from miniai.datasets import *
import torch.nn.functional as F

x,y = 'image','label'
name = 'fashion_mnist'
dsr = load_dataset(name)
dsr

@inplace
def transformi(b):
    b[x] = [torch.flatten(TF.to_tensor(i)) for i in b[x]]

dsrt = dsr.with_transform(transformi)

bs = 50
dls = DataLoaders.from_dd(dsrt, batch_size=bs)


xb,yb = next(iter(dls.train))
xb.shape,yb.shape

from torch import optim
import fastcore.all as fc

class Learner:
    def __init__(self, model, dls, lr, loss_func, opt_func=optim.SGD):
        fc.store_attr()
    
    def calc_stats(self):
        n = len(self.xb)
        self.accs.append((self.preds.argmax()==self.yb).float().sum())
        self.losses.append(self.loss*n)
        self.ns.append(n)
        
    def one_batch(self):
        self.xb,self.yb = self.batch
        self.preds = self.model(self.xb)
        self.loss = self.loss_func(self.preds, self.yb)
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        self.calc_stats()
            
    def one_epoch(self, train):
        self.model.training = train
        self.dl = self.dls.train if train else self.dls.valid
        for self.batch in self.dl:
            self.one_batch()
        ns = sum(self.ns) or 1
        avg_acc = sum(self.accs).item()/ns
        avg_loss = sum(self.loss).item()/ns
        print(f'trian:{train}, acc:{avg_acc:.3}, loss:{avg_loss:.3}')
    
    def fit(self, n_epochs):
        self.n_epochs = n_epochs
        self.ns,self.accs,self.losses = [],[],[]
        self.opt = self.opt_func(self.model.parameters(), lr=self.lr)
        for self.epoch in range(self.n_epochs):
            self.one_epoch(True)

n,nh,nout = 28*28,50,10
model = nn.Sequential(nn.Linear(n,nh), nn.ReLU(), nn.Linear(nh, nout))

learner = Learner(model, dls, lr=0.1, loss_func=F.cross_entropy)

learner.fit(5)
