from PredRNN_Model import PredRNN
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch


input=torch.rand(1,10,1,100,100).cuda()
target=torch.rand(1,10,1,100,100).cuda()
class PredRNN_enc(nn.Module):
    def __init__(self):
        super(PredRNN_enc, self).__init__()
        self.pred1_enc=PredRNN(input_size=(100,100),
                input_dim=1,
                hidden_dim=[7,1],
                hidden_dim_m=[7,7],
                kernel_size=(7,7),
                num_layers=2,
                batch_first=True,
                bias=True).cuda()
    def forward(self,enc_input):
        _, layer_h_c, last_h_m, _ = self.pred1_enc(enc_input)
        return layer_h_c, last_h_m

class PredRNN_dec(nn.Module):
    def __init__(self):
        super(PredRNN_dec, self).__init__()
        self.pred1_dec=PredRNN(input_size=(100,100),
                input_dim=1,
                hidden_dim=[7,1],
                hidden_dim_m=[7,7],
                kernel_size=(7,7),
                num_layers=2,
                batch_first=True,
                bias=True).cuda()
        self.relu = nn.ReLU()
    def forward(self,dec_input,enc_hidden,enc_h_m):
        out, layer_h_c, last_h_m, _ = self.pred1_dec(dec_input,enc_hidden,enc_h_m)
        out = self.relu(out)
        return out, layer_h_c, last_h_m

enc=PredRNN_enc().cuda()
dec=PredRNN_dec().cuda()

import itertools
loss_fn=nn.MSELoss()
position=0
optimizer=optim.Adam(itertools.chain(enc.parameters(), dec.parameters()),lr=0.001)
for epoch in range(100):
    loss_total=0
    enc_hidden, enc_h_y = enc(input)
    for i in range(input.shape[1]):
        optimizer.zero_grad()
        out,layer_h_c,last_h_y = dec(input[:,i:i+1,:,:,:], enc_hidden, enc_h_y)
        loss=loss_fn(out,target[:,i:i+1,:,:,:])
        loss_total+=loss
        enc_hidden = layer_h_c
        enc_h_y = last_h_y
    loss_total=loss_total/input.shape[1]
    loss_total.backward()
    optimizer.step()
    print(epoch,epoch,loss_total)


input_test = a.cuda()
target_test = b.cuda()
enc_hidden_test, enc_h_y_test = enc(a[0:1,:10,:,:,:].cuda())
out_test = input_test[:, 0:0 + 1, :, :, :]
loss_total_test = 0
for i in range(0,20):
    out_test, layer_h_c_test, last_h_y_test = dec(out_test, enc_hidden_test, enc_h_y_test)
    loss_test = loss_fn(out_test, target_test[:, i:i + 1, :, :, :])
    print(loss_test)
    loss_total_test += loss_test
    enc_hidden_test = layer_h_c_test
    enc_h_y_test = last_h_y_test
    out = out_test.cpu()
    out = out.detach().numpy()
    plt.axis('off')
    plt.imshow(out[0, 0, 0, :, :],cmap='binary_r')
    plt.savefig('C:/Users/yaoyuye/Desktop/pytorch_work/PRED/' + str(i))
print(loss_total_test/input_test.shape[1])


for i in range(0,20):
    plt.axis('off')
    plt.imshow(b[0, i, 0, :, :],cmap='binary_r')
    plt.savefig('C:/Users/yaoyuye/Desktop/pytorch_work/ACTUAL/' + str(i))