import model

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

batch_size = 512
block_size = 32

train_data = np.load('./train_data.npz')['train_data']

def get_batch():
    data = torch.Tensor(train_data)
    ix = torch.randint(data.size(0), (512,))
    x = torch.stack([data[i,:].type(torch.int64) for i in ix])
    return x

resume = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
musicvae = model.MusicVAE(seq_len=block_size).to(device)

if resume:
    musicvae.load_state_dict(torch.load("/content/musicvae_checkpoint.pt"))

musicvae.train()

optimizer = optim.Adam(musicvae.parameters(), lr=0.001)

# define scaduler
def scheduler_func(iter):
    if iter < 1000:
        return 1
    else:
        return 0.1

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = scheduler_func)

criterion = nn.NLLLoss(reduction='mean') #nn.CrossEntropyLoss()

num_iter = 2000
loss_avg_interval = 10

loss_list = []
NLL_list = []
KLD_list = []

loss_record = [] #loss record will saved as loss_record.pickle

print("traning start")
print("batch_size : 512, max_seq_len : 32, num_iter : 2000")

for iter in range(num_iter):
    x = get_batch()
    x = x.to(device)

    prob, mu, sigma  = musicvae(x)

    NLL = criterion(prob.transpose(1,2),x)
    KLD = (0.5 * torch.mean(mu.pow(2) + sigma.pow(2) - 1 - sigma.pow(2).log()))
    loss = NLL + KLD


    if iter % loss_avg_interval == 0 and iter != 0:
        print(torch.argmax(prob[0][0],dim=-1))
        print(x[0,0])
        
        avg_loss = sum(loss_list)/loss_avg_interval
        avg_NLL = sum(NLL_list)/loss_avg_interval
        avg_KLD = sum(KLD_list)/loss_avg_interval

        loss_record.append((iter,avg_loss))

        print("iter : ", iter, "avg_loss : ", avg_loss, "avg_NLL : ", avg_NLL, "avg_KLD : ", avg_KLD)
        loss_list = []
        NLL_list = []
        KLD_list = []

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_list.append(loss.item())
    KLD_list.append(KLD.item())
    NLL_list.append(NLL.item())

import pickle

with open('/content/loss_record.pickle', 'wb') as f:
    pickle.dump(loss_record, f)

torch.save(musicvae.state_dict(),'./musicvae_checkpoint.pt')