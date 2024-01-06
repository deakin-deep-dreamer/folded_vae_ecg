
import torch
import torch.nn as nn
import torch.nn.functional as F


class Params:
    def __init__(self, data_dict):
        for k, v in data_dict.items():
            exec("self.%s=%s" % (k, v))


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=1, is_conv=True):
        super(Conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.is_conv = is_conv
        self.relu = nn.LeakyReLU()
        self.pool_op = nn.AvgPool1d(2, ) if is_conv \
            else nn.Upsample(scale_factor=2, mode='linear')
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, 
            padding=padding, stride=2 if is_conv else 1)
        if is_conv:
            self.pool_op = nn.Sequential()
            # self.relu = nn.ELU()
        # self.bn = nn.BatchNorm1d(out_channels, eps=0.001, momentum=0.99)
        # self.dropout = nn.Dropout(0.2)
        # initialise layer weights
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
    
    def forward(self, x):
        if not self.is_conv:
            x = self.pool_op(x)
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        # x = self.dropout(x)
        if self.is_conv:
            return self.pool_op(x)
        else:
            return x
        # return self.pool_op(x)


class FoldEncoder(nn.Module):
    def __init__(self, n_block=3, input_dim=64*4, vae_latent_dim=-1, kernel_sz=3, 
                 input_split=4, log=print, is_vae=False, is_cuda=True, is_gru=False,
                 out_chan=8,):
        super(FoldEncoder, self).__init__()
        self.iter = 0
        self.log = log
        self.input_split = input_split
        self.is_vae = is_vae
        self.is_gru = is_gru
        layers = []
        in_chan, out_chan = 1, out_chan
        self.bn0 = nn.BatchNorm1d(1, eps=0.001, momentum=0.99)
        cur_input_dim = input_dim // input_split

        # TODO: need to check if perfectly splitted?
        # 

        for _ in range(n_block):
            padding = kernel_sz//2
            layers += [
                Conv_block(in_chan, out_chan, kernel_sz, padding=padding),
            ]
            in_chan = out_chan
            cur_input_dim //= 2
        else:
            # revert back input_dim since updated but not applied
            self.final_out_dim = cur_input_dim
            self.final_out_chan = out_chan
            
        self.encoder = nn.Sequential(*layers)
        # self.latent_dim = input_dim//4  # latent dim length
        self.latent_dim = in_chan * cur_input_dim * input_split
        self.post_encoder = nn.Sequential(
            # nn.Linear(in_chan*cur_input_dim*input_split, self.latent_dim)
            # nn.Conv1d(in_chan, 1, kernel_size=1),
            # nn.LeakyReLU(inplace=True),
        )
        
        # VAE specific
        # 
        if is_vae:
            # for the gaussian likelihood
            self.log_scale = nn.Parameter(torch.Tensor([0.0]))
            
            hidden_dim = in_chan * cur_input_dim * input_split
            
            self.latent_dim = hidden_dim // 2  # latent dim length
            if vae_latent_dim > -1:
                self.latent_dim = vae_latent_dim
            
            self.layer_mu = nn.Linear(hidden_dim, self.latent_dim)
            self.layer_sigma = nn.Linear(hidden_dim, self.latent_dim)
            # self.N = torch.distributions.Normal(0, 1)
            # if is_cuda:
            #     self.N.loc = self.N.loc.cuda()
            #     self.N.scale = self.N.scale.cuda()
            self.kl = 0
            self.init_weights()

    def init_weights(self):
        self.layer_mu.weight.data.normal_(0, 0.01)
        self.layer_sigma.weight.data.normal_(0, 0.01)

    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = self.bn0(x)
        
        # split into input_split
        b, c, w = x.size()
        w_split = w // self.input_split
        out_cat = None
        for i_split in range(self.input_split):
            x_split = x[:, :, i_split*w_split:(i_split+1)*w_split]
            # self.debug(f"   [{i_split}] x_split:{x_split.shape}")
            _z = self.encoder(x_split)
            # self.debug(f"   [{i_split}] z:{_z.shape}")

            if out_cat is None:
                out_cat = _z
            else:
                out_cat = torch.cat((out_cat, _z), dim=2)
            self.debug(f"   [{i_split}] x_split:{x_split.size()}, z:{_z.size()}, out_cat:{out_cat.size()}")

        final_out = out_cat

        if self.is_vae:
            out = out_cat.view(out_cat.size(0), -1)
            self.debug(f'   flatten: {out.shape}')

            mu = self.layer_mu(out)
            self.debug(f"   mu: {mu.shape}")

            sigma = self.layer_sigma(out)
            sigma_exp = torch.exp(sigma)
            self.debug(f"   sigma: {sigma_exp.shape}")

            # z = mu + sigma_exp*self.N.sample(mu.shape)
            
            # Alt 1: sample z from q
            # q = torch.distributions.Normal(mu, sigma_exp)
            # z = q.rsample()
            # Alt 2
            epsilon = torch.distributions.Normal(0, 1).sample(sigma.shape).to(sigma.device)
            z = mu + sigma_exp * epsilon
            # z = mu + torch.exp(0.5*sigma) * epsilon

            self.debug(f"   z: {z.shape}")

            # self.kl = (sigma**2 + mu**2 - torch.log(sigma_exp) - 1/2).sum()
            # alt 1
            # self.kl = self.kl_divergence(z, mu, sigma_exp).mean()
            # alt 2
            self.kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), dim=1)
            self.kl = self.kl.mean()

            final_out = z

        self.iter += 1
        return final_out

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class FoldDecoder(nn.Module):
    def __init__(
            self, n_block=5, latent_dim=512, kernel_sz=3, input_dim=95, 
            input_chan=3, input_split=None, is_vae=False, log=print):
        super(FoldDecoder, self).__init__()
        self.iter = 0
        self.log = log
        # self.latent_dim = latent_dim
        self.input_chan = input_chan
        self.input_dim = input_dim
        self.input_split = input_split
        self.is_vae = is_vae
        # self.length_adjusted = length_adjusted
        layers = []
        out_chan = input_chan
        self.pre_decoder = nn.Sequential(
            nn.Linear(latent_dim, input_chan*input_dim*input_split),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
        )
        if not is_vae:
            # void pre_decoder if not vae.
            self.pre_decoder = nn.Sequential()

        for i_block in range(n_block):
            layers += [
                Conv_block(
                    input_chan, 
                    out_chan if i_block < n_block-1 else 1, 
                    kernel_sz, padding=kernel_sz//2, is_conv=False),
            ]
        self.decoder = nn.Sequential(*layers)
        self.post_decoder = nn.Sequential(
            # nn.Conv1d(input_chan, 1, kernel_size=1),
            # nn.Tanh(),
            nn.Sigmoid()
        ) 
    
    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = self.pre_decoder(x)
        self.debug(f"   pre-decoder: {x.shape}")

        if self.is_vae:
            x = x.view(x.size(0), self.input_chan, self.input_dim*self.input_split)
            self.debug(f"   reshape: {x.shape}")
        
        # split into input_split
        b, c, w = x.size()
        w_split = w // self.input_split
        out_cat = None
        for i_split in range(self.input_split):
            x_split = x[:, :, i_split*w_split:(i_split+1)*w_split]
            # self.debug(f"   [{i_split}] x_split:{x_split.shape}")
            _z = self.decoder(x_split)
            # self.debug(f"   [{i_split}] z:{_z.shape}")
            if out_cat is None:
                out_cat = _z
            else:
                out_cat = torch.cat((out_cat, _z), dim=2)
            self.debug(f"   [{i_split}] x_split:{x_split.size()}, z:{_z.size()}, out_cat:{out_cat.size()}")
        x = out_cat


        # x = self.decoder(x)        
        self.debug(f"   decoder: {x.shape}")

        x = self.post_decoder(x)
        self.debug(f"   post-decoder: {x.shape}")

        self.iter += 1
        return x
    
    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class FoldDecoderZeroSplit(nn.Module):
    def __init__(
            self, n_block=5, latent_dim=512, kernel_sz=3, input_dim=95, 
            input_chan=3, input_split=None, is_vae=False, log=print):
        super(FoldDecoderZeroSplit, self).__init__()
        self.iter = 0
        self.log = log
        # self.latent_dim = latent_dim
        self.input_chan = input_chan
        self.input_dim = input_dim
        self.input_split = input_split
        self.is_vae = is_vae
        # self.length_adjusted = length_adjusted
        layers = []
        out_chan = input_chan
        self.pre_decoder = nn.Sequential(
            nn.Linear(latent_dim, input_chan*input_dim*input_split),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )
        if not is_vae:
            # void pre_decoder if not vae.
            self.pre_decoder = nn.Sequential()

        for _ in range(n_block):
            layers += [
                Conv_block(
                    input_chan, out_chan, kernel_sz, 
                    padding=kernel_sz//2,
                    is_conv=False),
            ]
        self.decoder = nn.Sequential(*layers)
        self.post_decoder = nn.Sequential(
            nn.Conv1d(input_chan, 1, kernel_size=1),
            nn.Tanh(),
            # nn.Sigmoid()
        ) 
    
    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = self.pre_decoder(x)
        self.debug(f"   pre-decoder: {x.shape}")

        if self.is_vae:
            x = x.view(x.size(0), self.input_chan, self.input_dim*self.input_split)
            self.debug(f"   reshape: {x.shape}")
        
        x = self.decoder(x)        
        self.debug(f"   decoder: {x.shape}")

        x = self.post_decoder(x)
        self.debug(f"   post-decoder: {x.shape}")

        self.iter += 1
        return x

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class FoldAE(nn.Module):
    def __init__(
            self, input_dim=100, vae_latent_dim=-1, input_split=4, n_block=2, n_class=2, 
            kernel_sz=None, log=print, is_cuda=False, is_vae=True):
        super(FoldAE, self).__init__()
        self.output_dim = n_class
        
        self.encoder = FoldEncoder(
            input_dim=input_dim, vae_latent_dim=vae_latent_dim, n_block=n_block, input_split=input_split,
            kernel_sz=kernel_sz, log=log, is_cuda=is_cuda, is_vae=is_vae)
        
        self.decoder = FoldDecoder(
            n_block=n_block,
            input_split=input_split,
            latent_dim=self.encoder.latent_dim,
            input_chan=self.encoder.final_out_chan, 
            input_dim=self.encoder.final_out_dim,
            kernel_sz=kernel_sz, is_vae=is_vae,
            log=log)

        # self.decoder = FoldDecoderZeroSplit(
        #     n_block=n_block,
        #     input_split=input_split,
        #     latent_dim=self.encoder.latent_dim, 
        #     input_chan=self.encoder.final_out_chan, 
        #     input_dim=self.encoder.final_out_dim,
        #     kernel_sz=kernel_sz, is_vae=is_vae,
        #     log=log)

        # For gaussian likelyhood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2))
    
    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class VAEEncoderCNN(nn.Module):
    def __init__(
            self, n_block=5, input_dim=100, kernel_sz=3, log=print, is_cuda=False):
        super(VAEEncoderCNN, self).__init__()
        self.iter = 0
        self.log = log
        layers = []
        in_chan = 1
        out_chan = 64
        self.bn0 = nn.BatchNorm1d(1, eps=0.001, momentum=0.99)
        cur_input_dim = input_dim
        for _ in range(n_block):
            padding = kernel_sz//2  
            layers += [
                Conv_block(in_chan, out_chan, kernel_sz, padding=padding),
            ]
            in_chan = out_chan
            cur_input_dim //= 2
        else:
            # revert back input_dim since updated but not applied
            self.final_out_dim = cur_input_dim
            self.final_out_chan = out_chan
            
        self.encoder = nn.Sequential(*layers)
        self.latent_dim = min(1024, in_chan*cur_input_dim//2)  # latent dim length
        self.post_encoder = nn.Sequential(
            # nn.Linear(in_chan*cur_input_dim, self.latent_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
        )

        # VAE specific
        # 
        # self.layer_mu = nn.Linear(self.latent_dim, self.latent_dim)
        # self.layer_sigma = nn.Linear(self.latent_dim, self.latent_dim)
        # self.N = torch.distributions.Normal(0, 1)
        # if is_cuda:
        #     self.N.loc = self.N.loc.cuda()
        #     self.N.scale = self.N.scale.cuda()
        #     self.kl = 0
    
    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = self.bn0(x)
        
        x = self.encoder(x)
        self.debug(f"   encoder: {x.shape}")

        x = x.view(x.size(0), -1)
        self.debug(f'   flatten: {x.shape}')

        x = self.post_encoder(x)
        self.debug(f'   post-enc: {x.shape}')

        # mu = self.layer_mu(x)
        # self.debug(f"   mu: {mu.shape}")

        # sigma = torch.exp(self.layer_sigma(x))
        # self.debug(f"   sigma: {sigma.shape}")

        # z = mu + sigma*self.N.sample(mu.shape)
        # self.debug(f"   z: {z.shape}")

        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        self.iter += 1
        return x  #z

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class VAEDecoderCNN(nn.Module):
    def __init__(
            self, n_block=5, latent_dim=512, kernel_sz=3, input_dim=95, 
            input_chan=3, log=print):
        super(VAEDecoderCNN, self).__init__()
        self.iter = 0
        self.log = log
        self.latent_dim = latent_dim
        self.input_chan = input_chan
        self.input_dim = input_dim
        # self.length_adjusted = length_adjusted
        layers = []
        out_chan = input_chan
        self.pre_decoder = nn.Sequential(
            # nn.Linear(latent_dim, input_chan*input_dim),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
        )
        for _ in range(n_block):
            layers += [
                Conv_block(
                    input_chan, out_chan, kernel_sz, 
                    padding=kernel_sz//2,
                    is_conv=False),
            ]
        self.decoder = nn.Sequential(*layers)
        self.post_decoder = nn.Sequential(
            nn.Conv1d(input_chan, 1, kernel_size=1),
            nn.Tanh(),
            # nn.Sigmoid()
        ) 
    
    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = self.pre_decoder(x)
        self.debug(f"   pre-decoder: {x.shape}")

        # x = x.view(x.size(0), self.in_chan, self.length_adjusted)
        x = x.view(x.size(0), self.input_chan, self.input_dim)
        self.debug(f"   reshape: {x.shape}")
        
        x = self.decoder(x)        
        self.debug(f"   decoder: {x.shape}")

        x = self.post_decoder(x)
        self.debug(f"   post-decoder: {x.shape}")

        self.iter += 1
        return x

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class VAECnn(nn.Module):
    def __init__(
            self, input_dim=100, n_class=2, kernel_sz=None, log=print, is_cuda=True):
        super(VAECnn, self).__init__()
        self.output_dim = n_class
        self.encoder = VAEEncoderCNN(
            input_dim=input_dim, 
            kernel_sz=kernel_sz, 
            is_cuda=is_cuda, log=log)
        self.decoder = VAEDecoderCNN(
            latent_dim=self.encoder.latent_dim, 
            input_chan=self.encoder.final_out_chan, 
            input_dim=self.encoder.final_out_dim,
            kernel_sz=kernel_sz, 
            log=log)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.encoder.latent_dim, self.encoder.latent_dim//2),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(self.encoder.latent_dim//2, n_class),
        #     nn.Softmax(dim=1),
        # )

        # For gaussian likelyhood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2))

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


class GruEncoder(nn.Module):
    def __init__(self, params):
        super(GruEncoder, self).__init__()
        
        self.params = params
        self.pre_encoder = nn.Sequential(
            nn.Conv1d(
                params.input_shape[0], params.input_shape[0], kernel_size=5, 
                dilation=2, padding='same'),
            nn.LeakyReLU(),
        )
        self.gru = nn.GRU(
            params.gru_n_features, 
            params.gru_hidden_dim, 
            num_layers=params.gru_num_layers,
            bidirectional=params.gru_num_directions==2,
            dropout=0.2,
            batch_first=True
        )
        self.hidden = None
        self._init_weights()
    
    def forward(self, x):
        batch_sz = x.shape[0]
        
        print(f"input: {x.shape}") if self.params.debug else None
        x = self.pre_encoder(x)
        print(f"pre_encoder: {x.shape}") if self.params.debug else None
        
        # x = x.view(x.size(0), -1)
        # print(f"reshape: {x.shape}") if self.params.debug else None
        
        _gru_out, self.hidden = self.gru(x, self.hidden)
        print(f"gru-h: {self.hidden.shape}, gru-out:{_gru_out.shape}") if self.params.debug else None
        
        # # hidden.shape = (n_layers * n_directions, batch_size, hidden_dim)
        # _h = self.hidden.view(
        #     self.params.gru_num_layers, self.params.gru_num_directions, 
        #     batch_sz, self.params.gru_hidden_dim)
        # # hidden.shape = (n_layers, n_directions, batch_size, hidden_dim)
        # hidden = hidden[-1]
        # # hidden.shape = (n_directions, batch_size, hidden_dim)
        # hidden_forward, hidden_backward = hidden[0], hidden[1]
        # # Both shapes (batch_size, hidden_dim)
        # fc_input = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        x = self.hidden.transpose(0, 1).contiguous().view(batch_sz, -1)
        print(f"gru-h-reshape: {x.shape}") if self.params.debug else None
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def init_hidden(self, batch, device=None):
        w = next(self.parameters()).data
        h = w.new(self.params.gru_num_layers*self.params.gru_num_directions, batch, self.params.gru_hidden_dim).zero_().to(device)
        return h

    def _sample(self, mean, logv):
        std = torch.exp(0.5 * logv)
        # torch.randn_like() creates a tensor with values samples from N(0,1) and std.shape
        eps = torch.randn_like(std)
        # Sampling from Z~N(μ, σ^2) = Sampling from μ + σX, X~N(0,1)
        z = mean + std * eps
        return z


class LSTMEncoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(LSTMEncoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    # x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    # print(f"[Encoder] rnn1 out:{x.shape}")
    x, (hidden_n, _) = self.rnn2(x)
    # print(f"[Encoder] rnn2 out:{x.shape}, h_n:{hidden_n.shape}")

    return hidden_n.permute(1, 0, 2)
    # return hidden_n.reshape((self.n_features, self.embedding_dim))


class LSTMDecoder(nn.Module):
  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(LSTMDecoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    # x = x.repeat(self.seq_len, self.n_features)
    # x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    # print(f"[Decoder] rnn1 out:{x.shape}")
    x, (hidden_n, cell_n) = self.rnn2(x)
    # print(f"[Decoder] rnn2 out:{x.shape}")
    # x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)
  
  
class RecurrentAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=64, is_debug=False):
    super(RecurrentAutoencoder, self).__init__()
    self.is_debug = is_debug
    self.encoder = LSTMEncoder(seq_len, n_features, embedding_dim)
    self.decoder = LSTMDecoder(seq_len, embedding_dim, n_features)

  def forward(self, x):
    print(f"[AE] input:{x.shape}") if self.is_debug else None
    x = self.encoder(x)
    print(f"[AE] enc out:{x.shape}") if self.is_debug else None
    x = self.decoder(x)
    print(f"[AE] dec out:{x.shape}") if self.is_debug else None
    return x
  


class VariationalEncoder_MLP512(nn.Module):
    def __init__(self, input_dim=None, latent_dim=None, is_cuda=False, log=print) -> None:
        super(VariationalEncoder_MLP512, self).__init__()
        self.iter = 0
        self.log = log
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, latent_dim)
        self.linear3 = nn.Linear(512, latent_dim)

        self.N = torch.distributions.Normal(0, 1)
        if is_cuda:
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
            self.kl = 0
    
    def forward(self, x):
        self.debug(f"input: {x.shape}")
        
        x = torch.flatten(x, start_dim=1)
        self.debug(f"flatten: {x.shape}")
        
        x = F.relu(self.linear1(x))
        self.debug(f"linear1: {x.shape}")

        mu = self.linear2(x)
        self.debug(f"mu: {mu.shape}")

        sigma = torch.exp(self.linear3(x))
        self.debug(f"sigma: {sigma.shape}")

        z = mu + sigma*self.N.sample(mu.shape)
        self.debug(f"z: {z.shape}")

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.iter += 1
        return z

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")

class Decoder_MLP512(nn.Module):
    def __init__(self, latent_dim=None, output_dim=None, log=print) -> None:
        super(Decoder_MLP512, self).__init__()
        self.iter = 0
        self.log = log
        self.output_dim = output_dim
        self.linear1 = nn.Linear(latent_dim, 512)
        self.linear2 = nn.Linear(512, output_dim)

    def forward(self, z):
        self.debug(f"input: {z.shape}")
        z = F.relu(self.linear1(z))
        self.debug(f"linear1: {z.shape}")
        z = torch.sigmoid(self.linear2(z))
        self.debug(f"linear2: {z.shape}")
        self.iter += 1
        return z.reshape(-1, 1, self.output_dim)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class VariationalAutoEncoder_MLP512(nn.Module):
    def __init__(self, input_dim=None, latent_dim=None, n_class=2, log=print):
        super(VariationalAutoEncoder_MLP512, self).__init__()
        self.output_dim = n_class
        self.encoder = VariationalEncoder_MLP512(
            input_dim=input_dim, latent_dim=latent_dim, log=log)
        self.decoder = Decoder_MLP512(
            latent_dim=latent_dim, output_dim=input_dim, log=log)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim//2, n_class),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z), self.classifier(z)



def test_vae_512(log=print):
    net = VariationalAutoEncoder_MLP512(input_dim=256, latent_dim=128)
    print(net)
    x = torch.randn(32, 1, 256)
    z, x_hat, out_clf = net(x)
    print(f"out:{x_hat.shape}")


def test_LstmAE():
    input_dim = 64*30
    batch_sz = 32
    x = torch.randn(batch_sz, 1, input_dim)

    net = RecurrentAutoencoder(seq_len=1, n_features=input_dim, embedding_dim=128, is_debug=True)
    print(net)
    net.hidden = net.init_hidden(batch_sz, 'cpu')
    out = net (x)
    print(f"decoded:{out.shape}")


def test_SqueezedAE():
    input_dim = 64*30
    x = torch.randn(32, 1, input_dim)
    net = FoldAE(
        input_dim=input_dim, input_split=8, n_block=4, kernel_sz=5, 
        is_cuda=False, is_vae=True)
    print(net)
    z, x_hat = net(x)
    print(z.shape, x_hat.shape)


def test_VAEEncoderCNN():
    input_dim = 64*30
    hidden_dim = 3840
    kernel_sz = 21
    x = torch.randn(32, 1, input_dim)
    
    # net = VAEEncoderCNN(input_dim=input_dim, kernel_sz=kernel_sz)
    # print(net)
    # out = net(x)
    # print(f"out:{out.shape}")

    # net = VAEDecoderCNN(
    #     latent_dim=hidden_dim, kernel_sz=kernel_sz, in_chan=64)
    # print(net)
    # out = net(out)
    # print(f"out:{out.shape}")

    net = VAECnn(input_dim=input_dim, n_class=2, kernel_sz=21, is_cuda=False)
    print(net)
    z, x_hat, out_clsf = net(x)
    print(f"z:{z.shape}, x_hat:{x_hat.shape}")


def test_GruAE():
    input_shape = (3, 64*30)
    batch_sz = 32
    x = torch.randn(batch_sz, input_shape[0], input_shape[-1])
    params = Params({
        'gru_n_features': input_shape[-1],
        'gru_hidden_dim': input_shape[-1] // 2,
        'gru_num_layers': 1,
        'gru_num_directions': 2,
        'input_shape': input_shape,
        'debug': True,
    })
    net = GruEncoder(params)
    print(net)
    net.hidden = net.init_hidden(batch_sz, 'cpu')
    out = net(x)
    print(out.shape)


if __name__ == "__main__":
    # test_SqueezedAE()
    test_GruAE()