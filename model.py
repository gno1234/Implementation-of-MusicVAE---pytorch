"""
# Paper: The content after the mark is what is mentioned in the text of the paper.  The model construction and parameter settings were based on the paper.
# https://arxiv.org/abs/1803.05428
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'


#################
#### Encoder ####
#################

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        ############parameter###################
        self.vocab_size = 258
        self.num_layers = 2
        self.hidden_size = 2048
        self.latent_size = 512
        
        
        ############encoder###################
        self.encoder_lstm = nn.LSTM(input_size=self.vocab_size,         # Paper : two-layer bidirectional LSTM network, LSTM state size of 2048 for all layers and 512 latent dimensions
                                    hidden_size=self.hidden_size,
                                    num_layers = self.num_layers,
                                    bidirectional=True)


        self.fc_mu = nn.Linear(self.hidden_size*2, self.latent_size)                            
        self.fc_sigma = nn.Linear(self.hidden_size*2, self.latent_size)
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # Paper :
        # We process an input sequence x = {x1, x2, . . . , xT } to obtain the final state vectors
        # →hT , ←hT from the second bidirectional LSTM layer. These are then concatenated to produce hT
        _, (h_n, _) = self.encoder_lstm(x) # in x=(batch, seq_len, n_embed) out x=(batch, seq_len, 2*hidden_size), h_n=(2*num_layers,batch,hidden_size)
        
        h_n = h_n[2:4,:,:] #last layer final output bidirectional , stacked
        h_n = h_n.reshape(1,h_n.size(1),self.hidden_size*2) #last layer final output bidirectional , concatenated

        mu = self.fc_mu(h_n)                     # Paper : fed into two fullyconnected layers to produce the latent distribution parameters µ and σ -> get "µ" : Equation (6)
                
        sigma = self.fc_sigma(h_n)               # Paper : fed into two fullyconnected layers to produce the latent distribution parameters µ and σ -> get "σ" : Equation (7)
        sigma = self.softplus(sigma)  # Paper : equation (7) # softplus activation function

        z = self.reparameterize(mu, sigma)       # Paper : As is standard in VAEs, µ and σ then parametrize the latent distribution as in Eq. (2)

        return z, mu, sigma                  # Encoder output

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std) # normal distribution with mean 0 and variance 1
        return mu + torch.mul(sigma,eps) #Hadamard product (element-wise product) # Paper equation (2)
    

#################
#### Decoder ####
#################

class Decoder(nn.Module):
    def __init__(self, seq_len = 32, batch_size = 512):
        super().__init__()
        #### Parameter ####
        self.batch_size = batch_size

        self.seq_len = seq_len
        self.sub_seq_len = 16
        self.U = int(self.seq_len/self.sub_seq_len)

        self.vocab_size =258

        self.latent_size = 512

        self.num_layers = 2
        self.conductor_hidden = 1024
        self.conductor_output_size = 512
        self.decoder_hidden = 1024


        ##### conductor #####
        self.conductor_initial_state = nn.Sequential(nn.Linear(self.latent_size,self.conductor_hidden), 
                                                     nn.Tanh())
        self.conductor_lstm = nn.LSTM(input_size =self.vocab_size, # Paper : we use a two-layer unidirectional LSTM for the conductor with a hidden state size of 1024 and 512 output dimensions
                                      hidden_size = self.conductor_hidden,
                                      num_layers = self.num_layers)
        self.conductor_output = nn.Linear(self.conductor_hidden,self.conductor_output_size)

        ##### decoder #####
        self.decoder_initial_state = nn.Sequential(nn.Linear(self.conductor_output_size, self.decoder_hidden),
                                                   nn.Tanh())
        self.decoder_lstm = nn.LSTM(input_size =self.conductor_output_size + self.vocab_size, # Paper : we used a 2-layer LSTM with 1024 units(hidden state) per layer for the decoder RNN
                                    hidden_size = self.decoder_hidden,
                                    num_layers = self.num_layers)
        ##### head ####
        self.fc_head = nn.Linear(self.decoder_hidden,self.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        ##### sos #####
        self.sos = F.one_hot(torch.zeros((1,self.batch_size),dtype=torch.int64,device=device),num_classes = self.vocab_size) #start token

        #### initial values ####
        self.conductor_input = nn.Parameter(torch.zeros((1,self.batch_size,self.vocab_size),requires_grad=True))


    def forward(self, z, x = None, teacher_forcing = True): # Training : teacher forcing = True , generating : teacher forcing = False
        # Paper :
        # the latent vector z is passed through a fully-connected layer4
        # followed by a tanh activation to get the initial state of a “conductor” RNN.
        conductor_initial_hidden_state_layer1 = self.conductor_initial_state(z)
        conductor_initial_state = self.initialize_state(conductor_initial_hidden_state_layer1)

        conductor_input = self.conductor_input.expand(self.U,self.batch_size,self.vocab_size)

        c, _ = self.conductor_lstm(conductor_input,conductor_initial_state) # Paper : The conductor RNN produces U embeddins vectors c = {c1, c2, . . . , cU }, one for each subsequence.
        c = self.conductor_output(c)

        # Paper :
        # Once the conductor has produced the sequence of embedding vectors c, 
        # each one is individually passed through a shared fully-connected layer 
        # followed by a tanh activation to produce initial states for a final 
        # bottom-layer decoder RNN
        decoder_initial_hidden_state_layer1= self.decoder_initial_state(c[0].unsqueeze(0))
        decoder_initial_state = self.initialize_state(decoder_initial_hidden_state_layer1)

        decoder_initial_input = torch.cat((c[0].unsqueeze(0),self.sos),dim =-1)

        probs = torch.zeros((0,self.batch_size,self.vocab_size),device=device)
        
        # Paper :
        # The decoder RNN then autoregressively produces a sequence of distributions 
        # over output tokens for each subsequence yu via a softmax output layer.
        for i in range(self.seq_len):

            j = i // self.sub_seq_len

            if i == 0:
                decoder_input = decoder_initial_input
                decoder_state = decoder_initial_state

            if i % self.sub_seq_len == 0 and i != 0:
                decoder_state = self.initialize_state(self.decoder_initial_state(c[j].unsqueeze(0)))
            
            if i != 0:
                # Paper : the current conductor embedding cu is concatenated with the previous output token to be used as the input.
                if teacher_forcing:
                    decoder_input = torch.cat((c[j].unsqueeze(0),x[i-1].unsqueeze(0)),dim=-1)
                else:
                    decoder_input = torch.cat((c[j].unsqueeze(0),decoder_output),dim=-1)

            
            decoder_output, decoder_state = self.decoder_lstm(decoder_input, decoder_state)

            decoder_output = self.softmax(self.fc_head(decoder_output))

            prob = torch.log(decoder_output)

            probs = torch.cat((probs,prob),dim = 0)

        return probs.permute(1,0,2)
            
            
    
    def initialize_state(self, hidden_state_layer1):
        hidden_state_layer2 = torch.zeros_like(hidden_state_layer1)
        hidden_state = torch.cat((hidden_state_layer1,hidden_state_layer2),dim=0)

        cell_state = torch.zeros_like(hidden_state)

        state = (hidden_state,cell_state)

        return state



    

##################
#### MusicVAE ####
##################

class MusicVAE(nn.Module):
    def __init__(self, teacher_forcing = True, seq_len = 32, batch_size =512):
        super().__init__()

        self.vocab_size = 258

        self.teacher_forcing = teacher_forcing
        
        self.Encoder = Encoder()
        self.Decoder = Decoder(teacher_forcing = teacher_forcing, seq_len = seq_len, batch_size = batch_size)

    def forward(self, x):

        x = F.one_hot(x,self.vocab_size).type(torch.float).permute(1,0,2)

        z, mu, sigma = self.Encoder(x)

        prob = self.Decoder(z, x, teacher_forcing = self.teacher_forcing)

        return prob, mu, sigma
