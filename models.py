
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

import vae_model


class GradientReverse(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, scale=1.0):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)


def padding_same(kernel_size, dilation=1):
    return (kernel_size-1) * dilation


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class FoldEncodedClassifier(nn.Module):
    def __init__(self, n_block=3, input_dim=64*4, kernel_sz=3, n_class=2,
            input_split=4, log=print, is_cuda=False) -> None:
        super(FoldEncodedClassifier, self).__init__()
        self.iter = 0
        self.log = log
        self.output_dim = n_class
        self.encoder = vae_model.FoldEncoder(
            n_block=n_block, input_dim=input_dim, kernel_sz=kernel_sz,
            input_split=input_split, log=log, is_vae=False, is_cuda=is_cuda
        )
        self.post_encoder = nn.Sequential(
            # nn.Linear(in_chan*cur_input_dim*input_split, self.latent_dim)
            nn.Conv1d(self.encoder.in_chan, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            # nn.Linear(self.encoder.latent_dim, self.encoder.latent_dim//2),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(self.encoder.latent_dim//2, n_class),
            nn.Linear(self.encoder.final_out_dim, n_class),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = self.encoder(x)
        self.debug(f"   encoder: {x.shape}")

        x = self.post_encoder(x)
        self.debug(f"   post-encoder: {x.shape}")

        x = x.view(x.size(0), -1)
        self.debug(f'   flatten: {x.shape}')
        
        x = self.classifier(x)
        self.debug(f'   classif: {x.shape}')
        
        self.iter += 1
        return x

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class FoldAEClassifier(nn.Module):
    def __init__(self, n_block=3, input_dim=64*4, vae_latent_dim=-1, 
                 kernel_sz=3, n_class=2, input_split=4, log=print, 
                 is_cuda=False, is_vae=False) -> None:
        super(FoldAEClassifier, self).__init__()
        self.iter = 0
        self.log = log
        self.output_dim = n_class
        self.autoencoder = vae_model.FoldAE(
            n_block=n_block, input_dim=input_dim, vae_latent_dim=vae_latent_dim,
            kernel_sz=kernel_sz,
            input_split=input_split, log=log, is_vae=is_vae, is_cuda=is_cuda
        )
        # self.squeeze_layer = nn.Sequential(
        #     nn.Conv1d(self.autoencoder.encoder.final_out_chan, 1, kernel_size=1),
        #     nn.LeakyReLU(inplace=True),
        # )
        # self.tcn = TemporalConvNet(
        #     num_inputs=1, num_channels=[8, 16, 1], kernel_size=3
        # )
        self.classifier = nn.Sequential(
            nn.Linear(self.autoencoder.encoder.latent_dim, 240),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(256, 64),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            # nn.Linear(self.autoencoder.encoder.latent_dim, n_class),
            nn.Linear(240, n_class),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        z, x_hat = self.autoencoder(x)
        self.debug(f"   encoder: {z.shape}")

        # z = self.squeeze_layer(z)
        # self.debug(f"   post-encoder: {z.shape}")

        # z = self.tcn(z)
        # self.debug(f"   tcn: {z.shape}")

        out = z.view(z.size(0), -1)
        self.debug(f'   flatten: {out.shape}')

        out = self.classifier(out)
        self.debug(f'   classif: {out.shape}')
        
        self.iter += 1
        return out, z, x_hat

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")



class FoldAEGruClassifier(nn.Module):
    def __init__(self, n_block=3, input_dim=64*4, kernel_sz=3, n_class=2,
            input_split=4, log=print, is_cuda=False, is_vae=False,
            gru_n_layers=2, gru_hidden_dim=None) -> None:
        super(FoldAEGruClassifier, self).__init__()
        self.iter = 0
        self.log = log
        self.output_dim = n_class
        self.gru_n_layers = gru_n_layers
        self.autoencoder = vae_model.FoldAE(
            n_block=n_block, input_dim=input_dim, kernel_sz=kernel_sz,
            input_split=input_split, log=log, is_vae=is_vae, is_cuda=is_cuda
        )
        self.gru_hidden_dim = gru_hidden_dim
        if gru_hidden_dim is None:
            self.gru_hidden_dim = self.autoencoder.encoder.latent_dim        
        self.gru = nn.GRU(
            self.gru_hidden_dim, self.gru_hidden_dim, gru_n_layers, 
            batch_first=True, dropout=0.2)
        
        # self.gru_decoded_ecg = nn.GRU(
        #     input_dim, gru_hidden_dim, gru_n_layers, batch_first=True, dropout=0.2
        # )

        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(self.gru_hidden_dim, n_class),
            nn.Softmax(dim=1),
        )

    def init_hidden(self, batch, device):
        w = next(self.parameters()).data
        h = w.new(self.gru_n_layers, batch, self.gru_hidden_dim).zero_().to(device)
        return h

    def forward(self, x, h):
        self.debug(f"   input: {x.shape}, h:{h.shape}")
        z, x_hat = self.autoencoder(x)
        self.debug(f"   encoder: {z.shape}, x_hat:{x_hat.shape}")

        out = z.view(z.size(0), -1)
        self.debug(f'   flatten: {out.shape}')
        out = out.unsqueeze(1)

        out, h = self.gru(out, h)
        self.debug(f"   GRU out:{out.shape}, h:{h.shape}")
        out = self.relu(out[:, -1])
        self.debug(f"   GRU slice:{out.shape}")

        # out, h = self.gru_decoded_ecg(x_hat)
        # out = self.relu(out[:, -1])
        # self.debug(f"   GRU slice:{out.shape}")

        out = out.view(out.size(0), -1)  # flatten GRU out

        out = self.classifier(out)
        self.debug(f'   classif: {out.shape}')
        
        self.iter += 1
        return out, z, x_hat, h

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class FoldEncoderClassifier(nn.Module):
    def __init__(self, n_block=3, input_dim=64*4, kernel_sz=3, n_class=2,
            input_split=4, log=print, is_cuda=False, is_vae=False) -> None:
        super(FoldEncoderClassifier, self).__init__()
        self.iter = 0
        self.log = log
        self.output_dim = n_class
        self.encoder = vae_model.FoldEncoder(
            n_block=n_block, input_dim=input_dim, kernel_sz=kernel_sz,
            input_split=input_split, log=log, is_vae=is_vae, is_cuda=is_cuda)
        
        self.squeeze_layer = nn.Sequential(
            nn.Conv1d(self.encoder.final_out_chan, 1, kernel_size=1),
            nn.LeakyReLU(inplace=True),
        )
        self.tcn = TemporalConvNet(
            num_inputs=1, num_channels=[8, 16, 1], kernel_size=3
        )
        self.classifier = nn.Sequential(
            # nn.Linear(self.autoencoder.encoder.latent_dim, 256),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # nn.Linear(256, 64),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            # # nn.Linear(self.autoencoder.encoder.final_out_dim*self.autoencoder.encoder.input_split, n_class),
            nn.Linear(self.encoder.final_out_dim*self.encoder.input_split, n_class),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        z = self.encoder(x)
        self.debug(f"   encoder: {z.shape}")

        z = self.squeeze_layer(z)
        self.debug(f"   post-encoder: {z.shape}")

        z = self.tcn(z)
        self.debug(f"   tcn: {z.shape}")

        out = z.view(z.size(0), -1)
        self.debug(f'   flatten: {out.shape}')

        out = self.classifier(out)
        self.debug(f'   classif: {out.shape}')
        
        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class VAEEncoderCNN(nn.Module):
    def __init__(self, n_block=5, input_dim=100, latent_dim=384, kernel_sz=3, hidden_dim=384, log=print, is_cuda=False):
        super(VAEEncoderCNN, self).__init__()
        self.iter = 0
        self.log = log
        layers = []
        in_chan = 1
        out_chan = 8
        for i_block in range(n_block):
            padd = padding_same(kernel_sz)
            layers += [
                nn.Conv1d(in_chan, out_chan, kernel_size=kernel_sz, stride=2, padding=0),
                nn.BatchNorm1d(out_chan),
                nn.ReLU(),   
                # nn.MaxPool1d(kernel_size=2, stride=2) 
            ]
            in_chan = out_chan
            out_chan *= 2            
        out_chan //= 2
        self.encoder = nn.Sequential(*layers)
        self.post_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer_mu = nn.Linear(hidden_dim, latent_dim)
        self.layer_sigma = nn.Linear(hidden_dim, latent_dim)
        self.N = torch.distributions.Normal(0, 1)
        if is_cuda:
            self.N.loc = self.N.loc.cuda()
            self.N.scale = self.N.scale.cuda()
            self.kl = 0
    
    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = self.encoder(x)
        self.debug(f"   encoder: {x.shape}")
        x = x.view(x.size(0), -1)
        self.debug(f'   flatten: {x.shape}')

        x = self.post_encoder(x)
        self.debug(f'   post-enc: {x.shape}')

        mu = self.layer_mu(x)
        self.debug(f"   mu: {mu.shape}")

        sigma = torch.exp(self.layer_sigma(x))
        self.debug(f"   sigma: {sigma.shape}")

        z = mu + sigma*self.N.sample(mu.shape)
        self.debug(f"   z: {z.shape}")

        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        self.iter += 1
        return z

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class VAEDecoderCNN(nn.Module):
    def __init__(self, n_block=5, latent_dim=512, hidden_align_dim=95, output_dim=None, kernel_sz=3, in_chan=2, log=print):
        super(VAEDecoderCNN, self).__init__()
        self.iter = 0
        self.log = log
        self.latent_dim = latent_dim
        self.in_chan = in_chan
        layers = []
        out_chan = in_chan = latent_dim//in_chan
        for i_block in range(n_block):
            out_chan //= 2
            layers += [
                nn.ConvTranspose1d(
                    in_channels=in_chan, 
                    out_channels=1 if i_block==n_block-1 else out_chan, 
                    kernel_size=kernel_sz, 
                    stride=2,
                    padding=0,  #1 if i_block < n_block-2 else 0,
                    output_padding=0,  #1 if i_block < n_block-1 else 0,
                    bias=False),
                # nn.Sigmoid() if i_block==n_block-1 else nn.ReLU()
                nn.ReLU(),
            ]
            in_chan = out_chan
        out_chan //= 2          
        self.decoder = nn.Sequential(*layers)
        self.post_decoder = nn.Sequential(
            nn.Linear(hidden_align_dim, output_dim),
            nn.Sigmoid()
        ) 
    
    def forward(self, x):
        self.debug(f"   input: {x.shape}")
        x = x.view(x.size(0), self.latent_dim//self.in_chan, self.in_chan)
        self.debug(f"   reshape: {x.shape}")
        
        out = self.decoder(x)        
        self.debug(f"   decoder: {out.shape}")

        out = self.post_decoder(out)
        self.debug(f"   post-decoder: {out.shape}")

        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class VariationalAutoEncoderCNN(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256, latent_dim=256, n_class=2, log=print):
        super(VariationalAutoEncoderCNN, self).__init__()
        self.output_dim = n_class
        self.encoder = VAEEncoderCNN(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, log=log)
        self.decoder = VAEDecoderCNN(
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


class DANN_MLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.iter = 0
        self.input_dim = args[0]
        self.feat_dim = args[1]
        self.output_dim = args[2]
        self.log = print if kwargs.get('log') is None else kwargs.get('log')
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.feat_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim*2, self.feat_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim//2, self.feat_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim//2, self.output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        self.debug(f"input: {x.shape}")
        b, c, d = x.size()
        x_src, x_dst = x[:, :-1], x[:, 1:]
        self.debug(f"x_src, x_dst: {x_src.shape}, {x_dst.shape}")

        x_src = x_src.view(x_src.size(0), -1)
        x_dst = x_dst.view(x_dst.size(0), -1)

        src_enc = self.encoder(x_src)
        self.debug(f"src_enc: {src_enc.shape}")
        src_out = self.decoder(src_enc)
        self.debug(f"src_out: {src_out.shape}")

        with torch.no_grad():
            dst_enc = self.encoder(x_dst)
            self.debug(f"dst_enc: {dst_enc.shape}")
            dst_out = self.decoder(dst_enc)
            self.debug(f"dst_out: {dst_out.shape}")

        self.iter += 1
        return src_enc, src_out, dst_enc, dst_out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class CNN_Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.iter = 0
        self.n_block = args[0]
        self.log = print if kwargs.get('log') is None else kwargs.get('log')
        conv_layers = []
        in_chan = 1
        out_chan = 8
        for _ in range(1, args[0]+1):            
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_chan, out_channels=out_chan, kernel_size=5, padding="same", bias=False),
                nn.BatchNorm1d(out_chan),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            in_chan = out_chan
            out_chan = in_chan * 2
        out_chan //= 2
        self.out_chan = out_chan
        self.encoder = nn.Sequential(*conv_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, grad_reverse_active=False):
        self.debug(f"input: {x.shape}, grad_reverse:{grad_reverse_active}")

        if grad_reverse_active:            
            x = grad_reverse(x)

        src_enc = self.encoder(x)
        self.debug(f"src_enc: {src_enc.shape}")
        src_enc = self.gap(src_enc)
        src_enc = src_enc.view(src_enc.size(0), -1)
        self.debug(f"src_enc flat: {src_enc.shape}")

        self.iter += 1
        return src_enc

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class Linear_Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.iter = 0
        in_chan = args[0]
        self.out_chan = args[1]
        self.log = print if kwargs.get('log') is None else kwargs.get('log')
        self.decoder = nn.Sequential(
            nn.Linear(in_chan, in_chan//2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_chan//2, self.out_chan),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        self.debug(f"input: {x.shape}")
        out = self.decoder(x)
        self.debug(f"out: {out.shape}")
        self.iter += 1
        return out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class DANN_CNN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.iter = 0
        self.n_block = args[0]
        self.output_dim = args[1]
        # self.ref_sample = args[2]
        self.feat_dim = kwargs.get('feat_dim')
        self.log = print if kwargs.get('log') is None else kwargs.get('log')
        conv_layers = []
        in_chan = 1
        out_chan = 8
        for _ in range(1, args[0]+1):
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_chan, out_channels=out_chan, kernel_size=5, padding="same", bias=True),
                nn.BatchNorm1d(out_chan),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
            ]
            in_chan = out_chan
            out_chan = in_chan * 2        
        out_chan //= 2

        self.encoder = nn.Sequential(*conv_layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.post_encoder = nn.Sequential(
            nn.Linear(out_chan, self.feat_dim*2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feat_dim*2, self.feat_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(out_chan, out_chan//2),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.feat_dim, self.output_dim),
            nn.Softmax(dim=1),
        )
        self.debug(f"n_block:{args[0]}, output_dim:{args[1]},")

    def freeze_conv_layers(self, first_n_layers=3, freeze=True):
        i_conv = 0
        for layer in self.encoder.children():
            if i_conv >= first_n_layers:
                break
            if isinstance(layer, nn.Conv1d):
                i_conv += 1
                for param in layer.parameters():
                    param.requires_grad = (not freeze)
        self.debug(f"First {first_n_layers} conv-layers frozen:{freeze}.")

    def freeze_conv_layer_range(self, layer_ranges=(1, 3), freeze=True):
        i_conv = 0
        count_layer_frozen = 0
        for layer in self.encoder.children():
            if isinstance(layer, nn.Conv1d):
                i_conv += 1
                if i_conv >= layer_ranges[0] and i_conv <= layer_ranges[1]:
                    count_layer_frozen += 1
                    for param in layer.parameters():
                        param.requires_grad = (not freeze)
        assert count_layer_frozen == (layer_ranges[1]-layer_ranges[0]+1)
        self.debug(f"Conv-layers {layer_ranges} frozen:{freeze}.")

    def forward(self, x, grad_reverse_active=False):
        self.debug(f"input: {x.shape}, grad_reverse:{grad_reverse_active}")

        if grad_reverse_active:            
            x = grad_reverse(x)

        out_enc = self.encoder(x)
        self.debug(f"out_enc: {out_enc.shape}")

        out_enc = self.gap(out_enc)
        out_enc = out_enc.view(out_enc.size(0), -1)
        self.debug(f"out_enc flat: {out_enc.shape}")

        out_enc = self.post_encoder(out_enc)
        self.debug(f"post_enc: {out_enc.shape}")
        
        out = self.decoder(out_enc)
        self.debug(f"src_out: {out.shape}")

        self.iter += 1
        return out_enc, out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class EEGConv(nn.Module):
    def __init__(self, h=100, n_classes=2, log=print, is_debug=True) -> None:
        super().__init__()
        self.log = log
        self.is_debug = is_debug
        self.output_dim = n_classes
        
        self.block0 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=21, padding="same", bias=False),
            nn.BatchNorm1d(8),
            nn.PReLU(),
        )
        self.block_depthconv = nn.Sequential(
            nn.Conv1d(8, 8*2, kernel_size=11, padding="same", bias=False, groups=8),
            nn.BatchNorm1d(8*2),
            nn.PReLU(),
            nn.AvgPool1d(kernel_size=4, stride=4),
            nn.Dropout1d(p=0.2),
        )
        self.block_pointconv = nn.Sequential(
            nn.Conv1d(8*2, 8*2, kernel_size=17, padding="same", bias=False, groups=8),
            nn.Conv1d(8*2, 8*2, kernel_size=1, padding="same", bias=False),
            nn.BatchNorm1d(8*2),
            nn.PReLU(),
            nn.AvgPool1d(kernel_size=8, stride=8),
            nn.Dropout1d(p=0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(h, n_classes),
            nn.Softmax(dim=1),   
        )
    
    def forward(self, x):
        self.debug(f"   input: {x.shape}") if self.is_debug else None
        out = self.block0(x)
        self.debug(f"   block0: {out.shape}") if self.is_debug else None
        out = self.block_depthconv(out)
        self.debug(f"   block_depthconv: {out.shape}") if self.is_debug else None
        out = self.block_pointconv(out)
        self.debug(f"   block_pointconv: {out.shape}") if self.is_debug else None
        enc_out = out = out.view(out.size(0), -1)
        self.debug(f"   out flat: {out.shape}") if self.is_debug else None
        out = self.classifier(out)
        self.debug(f"   out: {out.shape}") if self.is_debug else None

        if self.is_debug:
            self.is_debug = not self.is_debug
        return enc_out, out

    def debug(self, args):
        self.log(f"[{self.__class__.__name__}] {args}")

    
class ClassicCNNv1(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.iter = 0
        self.n_block = args[0]
        self.output_dim = args[1]
        # self.ref_sample = args[2]
        self.feat_dim = kwargs.get('feat_dim')
        self.log = print if kwargs.get('log') is None else kwargs.get('log')
        conv_layers = []
        in_chan = 1
        init_out_chan = out_chan = 8
        for i_block in range(1, args[0]+1):
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_chan, out_channels=out_chan, kernel_size=11, stride=2, 
                    # padding="same", 
                    groups=1 if i_block==1 else init_out_chan,
                    bias=False
                ),
                # nn.BatchNorm1d(out_chan),
                nn.PReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2),
                # nn.Dropout(0.3),
            ]
            in_chan = out_chan
            out_chan = in_chan * 2
        out_chan //= 2

        self.bn0 = nn.BatchNorm1d(1)
        self.encoder = nn.Sequential(*conv_layers)
        gap_factor = 4
        self.gap = nn.AdaptiveAvgPool1d(gap_factor)
        self.post_encoder = nn.Sequential(
            nn.Linear(out_chan*gap_factor, self.feat_dim*2),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.feat_dim*2, self.feat_dim*2),
            nn.PReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.feat_dim*2, self.feat_dim),
            nn.PReLU(),
            # nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            # nn.Linear(out_chan, out_chan//2),
            # nn.LeakyReLU(),
            # nn.Dropout(0.2),
            nn.Linear(self.feat_dim, self.output_dim),
            # nn.Softmax(dim=1),
        )
        self.debug(f"n_block:{args[0]}, output_dim:{args[1]},")

    def freeze_conv_layers(self, first_n_layers=3, freeze=True):
        i_conv = 0
        for layer in self.encoder.children():
            if i_conv >= first_n_layers:
                break
            if isinstance(layer, nn.Conv1d):
                i_conv += 1
                for param in layer.parameters():
                    param.requires_grad = (not freeze)
        self.debug(f"First {first_n_layers} conv-layers frozen:{freeze}.")

    def freeze_conv_layer_range(self, layer_ranges=(1, 3), freeze=True):
        i_conv = 0
        count_layer_frozen = 0
        for layer in self.encoder.children():
            if isinstance(layer, nn.Conv1d):
                i_conv += 1
                if i_conv >= layer_ranges[0] and i_conv <= layer_ranges[1]:
                    count_layer_frozen += 1
                    for param in layer.parameters():
                        param.requires_grad = (not freeze)
        assert count_layer_frozen == (layer_ranges[1]-layer_ranges[0]+1)
        self.debug(f"Conv-layers {layer_ranges} frozen:{freeze}.")

    def forward(self, x, grad_reverse_active=False):
        self.debug(f"input: {x.shape}, grad_reverse:{grad_reverse_active}")

        if grad_reverse_active:            
            x = grad_reverse(x)

        x = self.bn0(x)

        out_enc = self.encoder(x)
        self.debug(f"out_enc: {out_enc.shape}")

        out_enc = self.gap(out_enc)
        out_enc = out_enc.view(out_enc.size(0), -1)
        self.debug(f"out_enc flat: {out_enc.shape}")

        out_enc = self.post_encoder(out_enc)
        self.debug(f"post_enc: {out_enc.shape}")
        
        out = self.classifier(out_enc)
        self.debug(f"src_out: {out.shape}")

        self.iter += 1
        return out_enc, out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")



class ClassicCNNv2(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.iter = 0
        self.n_block = args[0]
        self.output_dim = args[1]
        # self.ref_sample = args[2]
        self.feat_dim = kwargs.get('feat_dim')
        self.log = print if kwargs.get('log') is None else kwargs.get('log')
        conv_layers = []
        in_chan = 1
        init_out_chan = out_chan = 8
        for i_block in range(1, args[0]+1):
            conv_layers += [
                nn.Conv1d(
                    in_channels=in_chan, out_channels=out_chan, kernel_size=11, stride=2, 
                    # padding="same", 
                    groups=1 if i_block==1 else init_out_chan,
                    bias=False
                ),
                # nn.BatchNorm1d(out_chan),
                nn.PReLU(),
                # nn.MaxPool1d(kernel_size=2, stride=2),
                # nn.Dropout(0.3),
            ]
            in_chan = out_chan
            out_chan = in_chan * 2
        out_chan //= 2

        self.bn0 = nn.BatchNorm1d(1)
        self.encoder = nn.Sequential(*conv_layers)
        gap_factor = 4
        self.gap = nn.AdaptiveAvgPool1d(gap_factor)
        self.post_encoder = nn.Sequential(
            nn.Linear(out_chan*gap_factor, self.feat_dim),
            nn.PReLU(),
            # nn.Dropout(0.3),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.PReLU(),
            # nn.Dropout(0.3),
            nn.Linear(self.feat_dim, 2),
            nn.PReLU(),
            # nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2, self.output_dim),
            # nn.Softmax(dim=1),
        )
        self.debug(f"n_block:{args[0]}, output_dim:{args[1]},")

    def forward(self, x):
        self.debug(f"input: {x.shape}")

        x = self.bn0(x)

        out_enc = self.encoder(x)
        self.debug(f"out_enc: {out_enc.shape}")

        out_enc = self.gap(out_enc)
        out_enc = out_enc.view(out_enc.size(0), -1)
        self.debug(f"out_enc flat: {out_enc.shape}")

        out_enc = self.post_encoder(out_enc)
        self.debug(f"post_enc: {out_enc.shape}")
        
        out = self.classifier(out_enc)
        self.debug(f"src_out: {out.shape}")

        self.iter += 1
        return out_enc, out

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


def test_model():
    # net = DANN_MLP(100, 200, 2, log=print)
    net = DANN_CNN(5, 2, log=print)
    print(net)
    x = torch.randn(32, 2, 100*30)
    out = net(x)
    print(f"out {out[0].shape}, {out[1].shape}")



class Resnet(nn.Module):
    r"""Convolution network."""

    def __init__(
        self, segment_sz, kernels=None, dilations=None, in_channels=None, out_channels=None,
        conv_groups=None, n_conv_layers_per_block=1, n_blocks=2,
        n_classes=None, low_conv_options=None, shortcut_conn=False, log=print,
        gap_layer=True, hidden_neurons_classif=-1, adaptive_avg_len=1
    ):
        r"""Instance of convnet."""
        super(Resnet, self).__init__()

        log(
            f"segment_sz:{segment_sz}, kernels:{kernels}, in-chan:{in_channels}, "
            f"out-chan:{out_channels}, conv-gr:{conv_groups}, "
            f"n-conv-layer-per-block:{n_conv_layers_per_block}, "
            f"n_block:{n_blocks}, n_class:{n_classes}, "
            f"low-conv:{low_conv_options}, shortcut:{shortcut_conn}")

        self.iter = 0
        self.log = log
        self.input_sz = self.segment_sz = segment_sz
        self.kernels = kernels
        self.dilations = dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_groups = conv_groups
        self.n_conv_layers_per_block = n_conv_layers_per_block
        self.n_blocks = n_blocks
        self.low_conv_options = low_conv_options
        self.shortcut_conn = shortcut_conn
        self.gap_layer = gap_layer
        self.hidden_neurons_classif = hidden_neurons_classif
        self.output_dim = n_classes

        self.input_bn = nn.BatchNorm1d(self.in_channels)
        self.low_conv = self.make_low_conv()
        self.encoder = self.make_layers_deep()
        self.gap = nn.AdaptiveAvgPool1d(adaptive_avg_len)
        hidden_len = self._out_channels*adaptive_avg_len
        self.post_encoder = nn.Sequential(
            nn.Linear(hidden_len, hidden_len//2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(hidden_len//2, n_classes)

    def forward(self, x):
        self.debug(f'  input: {x.shape}')

        out = self.low_conv(x)
        self.debug(f'  low_conv out: {out.shape}')

        out = self.encoder(out)
        self.debug(f"  encoder out: {out.shape}")
        out = self.gap(out)
        self.debug(f"  GAP out: {out.shape}")

        out = out.view(out.size(0), -1)
        self.debug(f'  flatten: {out.shape}')
        out_enc = self.post_encoder(out)
        out = out_enc
        self.debug(f'  post-enc out: {out.shape}')
        out = self.classifier(out)
        self.debug(f'  out: {out.shape}')
        self.iter += 1
        return out_enc, out

    def calculate_hidden(self):
        return self.input_sz * self.out_channels[-1]

    def make_layers_deep(self):
        in_channels = self.low_conv_hidden_dim
        layers = []
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            layers_for_shortcut = []
            in_channel_for_shortcut = in_channels
            for _ in range(self.n_conv_layers_per_block):
                layers_for_shortcut += [
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        dilation=self.dilations[i],
                        padding="same",
                    ),
                    nn.BatchNorm1d(self._out_channels),
                    nn.LeakyReLU(inplace=False)
                ]
                in_channels = self._out_channels
            layers += [
                ShortcutBlock(
                        layers=nn.Sequential(*layers_for_shortcut),
                        in_channels=in_channel_for_shortcut,
                        out_channels=self._out_channels,
                        point_conv_group=self.conv_groups[i])
            ]
        return nn.Sequential(*layers)

    def make_low_conv(self):
        r"""Make low convolution block."""
        layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = self.in_channels
        for x in self.low_conv_options["cfg"]:
            if x == 'M':
                count_pooling += 1
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.input_sz /= 2
            else:
                stride_ = self.low_conv_options["stride"][i_kernel]
                conv = nn.Conv1d(
                    in_channels=in_chan, out_channels=x,
                    kernel_size=self.low_conv_options["kernel"][i_kernel],
                    groups=self.low_conv_options["conv_groups"][i_kernel],
                    stride=stride_,
                    dilation=self.low_conv_options["dilation"][i_kernel],
                )
                layers += [
                    conv,
                    nn.BatchNorm1d(x),
                    nn.LeakyReLU(inplace=False)
                    ]
                in_chan = self.low_conv_hidden_dim = x
                self.input_sz /= self.low_conv_options["stride"][i_kernel]
                i_kernel += 1
            pass    # for
        return nn.Sequential(*layers)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class ClassicConv(nn.Module):
    r"""Convolution network."""

    def __init__(
        self, segment_sz, kernels=None, dilations=None, in_channels=None, out_channels=None,
        conv_groups=None, n_conv_layers_per_block=1, n_blocks=2,
        n_classes=None, low_conv_options=None, shortcut_conn=False, log=print,
        gap_layer=True, hidden_neurons_classif=-1, adaptive_avg_len=1
    ):
        r"""Instance of convnet."""
        super(ClassicConv, self).__init__()

        log(
            f"segment_sz:{segment_sz}, kernels:{kernels}, in-chan:{in_channels}, "
            f"out-chan:{out_channels}, conv-gr:{conv_groups}, "
            f"n-conv-layer-per-block:{n_conv_layers_per_block}, "
            f"n_block:{n_blocks}, n_class:{n_classes}, "
            f"low-conv:{low_conv_options}, shortcut:{shortcut_conn}")

        self.iter = 0
        self.log = log
        self.input_sz = self.segment_sz = segment_sz
        self.kernels = kernels
        self.dilations = dilations
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_groups = conv_groups
        self.n_conv_layers_per_block = n_conv_layers_per_block
        self.n_blocks = n_blocks
        self.low_conv_options = low_conv_options
        self.shortcut_conn = shortcut_conn
        self.gap_layer = gap_layer
        self.hidden_neurons_classif = hidden_neurons_classif

        self.input_bn = nn.BatchNorm1d(self.in_channels)
        self.low_conv = self.make_low_conv()
        # self.features = self.make_layers_deep()

        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.pool = nn.ModuleList([])
        self.shortcut = nn.ModuleList([])
        self.make_layers_deep()
        if self.gap_layer:
            self.gap = nn.AdaptiveAvgPool1d(adaptive_avg_len)
            self.classifier = nn.Linear(self._out_channels*adaptive_avg_len, n_classes)
        else:
            self.classifier = nn.Linear(hidden_neurons_classif, n_classes)

    def name(self):
        return (
            f"{self.__class__.__name__}_"
            f"segsz{self.segment_sz}_scut{self.shortcut_conn}_"
            f"lccfg{'x'.join(str(x) for x in self.low_conv_options['cfg'])}_"
            f"lckr{'x'.join(str(x) for x in self.low_conv_options['kernel'])}_"
            f"lcst{'x'.join(str(x) for x in self.low_conv_options['stride'])}_"
            f"lccg{'x'.join(str(x) for x in self.low_conv_options['conv_groups'])}_"
            f"blk{self.n_blocks}_cpblk{self.n_conv_layers_per_block}_"
            f"kr{'x'.join(str(x) for x in self.kernels)}_"
            f"och{'x'.join(str(x) for x in self.out_channels)}_"
            f"cg{'x'.join(str(x) for x in self.conv_groups)}")
        # return f'{self.__class__.__name__}'

    def forward(self, x):
        self.debug(f'  input: {x.shape}')

        x = self.input_bn(x)

        out = self.low_conv(x)
        self.debug(f'  low_conv out: {out.shape}')

        for i_blk in range(self.n_blocks):
            if self.shortcut_conn:
                out = self.shortcut[i_blk](out)
                self.debug(f"[block:{i_blk}] shortcut out: {out.shape}")
            else:
                for i_conv_blk in range(self.n_conv_layers_per_block):
                    idx_flat = 2*i_blk+i_conv_blk
                    self.debug(
                        f"  block({i_blk}) conv({i_conv_blk}) {self.conv[idx_flat]}")
                    out = self.conv[idx_flat](out)
                    out = self.bn[idx_flat](out)
                    out = self.act[idx_flat](out)
                self.debug(
                    f"  block({i_blk}) out:{out.shape}, data:{out.detach().cpu()[0, 0, :10]}")
            r"One less pooling layer."
            if i_blk < self.n_blocks - 1:
                out = self.pool[i_blk](out)
                self.debug(f"  block({i_blk}) pool-out:{out.shape}")

        if self.gap_layer:
            out = self.gap(out)
            self.debug(f"  GAP out: {out.shape}")

        out = out.view(out.size(0), -1)
        self.debug(f'  flatten: {out.shape}')
        out = self.classifier(out)
        self.debug(f'  out: {out.shape}')
        self.iter += 1
        return out

    def calculate_hidden(self):
        return self.input_sz * self.out_channels[-1]

    def make_layers_deep(self):
        in_channels = self.low_conv_hidden_dim
        for i in range(self.n_blocks):
            self._out_channels = self.out_channels[i]
            layers_for_shortcut = []
            in_channel_for_shortcut = in_channels
            for _ in range(self.n_conv_layers_per_block):
                self.conv.append(
                    nn.Conv1d(
                        in_channels,
                        self._out_channels,
                        kernel_size=self.kernels[i],
                        groups=self.conv_groups[i],
                        # Disable bias in convolutional layers before batchnorm.
                        bias=False,
                        dilation=self.dilations[i],
                        padding="same",
                        # padding=padding_same(
                        #     # input=self.input_sz,
                        #     kernel=self.kernels[i],
                        #     # stride=1,
                        #     dilation=self.dilations[i])
                    ))
                self.bn.append(nn.BatchNorm1d(self._out_channels))
                self.act.append(nn.ReLU(inplace=True))
                if self.shortcut_conn:
                    layers_for_shortcut.extend([
                        self.conv[-1], self.bn[-1],
                    ])
                in_channels = self._out_channels

            if self.shortcut_conn:
                self.shortcut.append(
                    ShortcutBlock(
                        layers=nn.Sequential(*layers_for_shortcut),
                        in_channels=in_channel_for_shortcut,
                        out_channels=self._out_channels,
                        point_conv_group=self.conv_groups[i])
                )
            if i < self.n_blocks - 1:
                self.pool.append(nn.MaxPool1d(2, stride=2))
                self.input_sz //= 2
        # return nn.Sequential(*layers)        
        r"If shortcut_conn is true, empty conv, and bn module-list. \
        This may be necessary to not to calculate gradients for the \
        same layer twice."
        if self.shortcut_conn:
            self.conv = nn.ModuleList([])
            self.bn = nn.ModuleList([])
            self.act = nn.ModuleList([])

    def make_low_conv(self):
        r"""Make low convolution block."""
        layers = []
        count_pooling = 0
        i_kernel = 0
        in_chan = self.in_channels
        for x in self.low_conv_options["cfg"]:
            if x == 'M':
                count_pooling += 1
                layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
                self.input_sz /= 2
            else:
                stride_ = self.low_conv_options["stride"][i_kernel]
                conv = nn.Conv1d(
                    in_channels=in_chan, out_channels=x,
                    kernel_size=self.low_conv_options["kernel"][i_kernel],
                    groups=self.low_conv_options["conv_groups"][i_kernel],
                    stride=stride_,
                    dilation=self.low_conv_options["dilation"][i_kernel],
                    padding="same",
                    # padding=padding_same(
                    #     # input=self.input_sz,
                    #     kernel=self.low_conv_options["kernel"][i_kernel],
                    #     # stride=1,
                    #     dilation=self.low_conv_options["dilation"][i_kernel]),
                )
                layers += [
                    conv,
                    nn.BatchNorm1d(x),
                    nn.ReLU(inplace=True)
                    ]
                in_chan = self.low_conv_hidden_dim = x
                self.input_sz /= self.low_conv_options["stride"][i_kernel]
                i_kernel += 1
            pass    # for
        return nn.Sequential(*layers)

    def debug(self, args):
        if self.iter == 0:
            self.log(f"[{self.__class__.__name__}] {args}")


class ShortcutBlock(nn.Module):
    """Pass a Sequence and add identity shortcut, following a ReLU."""

    def __init__(
        self,
        layers=None,
        in_channels=None,
        out_channels=None,
        point_conv_group=1,
        log=print,
    ):
        super(ShortcutBlock, self).__init__()
        self.iter = 0
        self.log = log
        self.layers = layers
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                    groups=point_conv_group,
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        # self.debug(f'input: {x.shape}')
        out = self.layers(x)
        # self.debug(f'layers out: {out.shape}')
        out += self.shortcut(x)
        # self.debug(f'shortcut out: {out.shape}')
        out = F.relu(out)
        self.iter += 1
        return out

    def debug(self, *args):
        if self.iter == 0:
            self.log(self.__class__.__name__, args)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def test_tcn():
    net = TemporalConvNet(
        num_inputs=1, num_channels=[8, 16, 1], kernel_size=3
    )
    print(net)
    x = torch.randn(32, 1, 30*64)
    out = net(x)
    print(f"out:{out.shape}")


def test_resnet(seg_sec=30, in_chan=1, n_block=5, n_classes=2, log=print):
    net = Resnet(
        segment_sz=10*seg_sec, shortcut_conn=True,
        in_channels=in_chan,
        kernels=[3, 3, 3, 3, 3],
        dilations=[1, 1, 1, 1, 1],
        out_channels=[64, 64, 64, 64, 64],
        conv_groups=[1 for i in range(n_block)],
        n_conv_layers_per_block=2, n_blocks=n_block, n_classes=n_classes,
        low_conv_options={
            "cfg": [64, 'M'],
            'stride': [2,],
            "kernel": [1,],
            'conv_groups': [in_chan,],
            'dilation': [1,],
            },
        log=log)
    print(net)
    x = torch.randn(32, 1, 30*10)
    out = net(x)
    print(f"out: {out.shape}")


def test_classic_conv(seg_sec=30, in_chan=1, n_block=5, n_classes=2, log=print):
    net = ClassicConv(
        segment_sz=10*seg_sec, shortcut_conn=True,
        in_channels=in_chan,
        kernels=[3, 3, 3, 3, 3],
        dilations=[1, 1, 1, 1, 1],
        out_channels=[32, 64, 128, 256, 512],
        conv_groups=[1 for i in range(n_block)],
        n_conv_layers_per_block=2, n_blocks=n_block, n_classes=n_classes,
        low_conv_options={
            "cfg": [8, 8, 'M'],
            'stride': [1, 1],
            "kernel": [1, 1],
            'conv_groups': [in_chan, in_chan],
            'dilation': [1, 1],
            },
        log=log)
    print(net)
    x = torch.randn(32, 1, 30*10)
    out = net(x)
    print(f"out: {out.shape}")


def test_EEGConv():
    net = EEGConv(h=960, n_classes=2)
    print(net)
    x = torch.randn(32, 1, 64*30)
    _, out = net(x)
    print(f"out:{out.shape}")


def test_classicCNN(n_block=5, n_classes=2, feat_dim=256, log=print):
    net = ClassicCNNv2(5, n_classes, feat_dim=feat_dim)
    print(net)
    x = torch.randn(32, 1, 64*30)
    out_enc, out = net(x)
    print(f"out_enc: {out_enc.shape}, out:{out.shape}")


def test_vae_512(log=print):
    net = VariationalAutoEncoder_MLP512(input_dim=100, latent_dim=128)
    print(net)
    x = torch.randn(32, 1, 100)
    out = net(x)
    print(f"out:{out.shape}")

def test_VAEEncoderCNN():
    x = torch.randn(32, 1, 100)
    
    # net = VAEEncoderCNN(input_dim=100, hidden_dim=256, latent_dim=256)
    # print(net)
    # out = net(x)
    # print(f"out:{out.shape}")

    # net = VAEDecoderCNN(latent_dim=256)
    # print(net)
    # out = net(out)
    # print(f"out:{out.shape}")

    net = VariationalAutoEncoderCNN(input_dim=100, hidden_dim=256, latent_dim=256)
    print(net)
    z, x_hat, out_clsf = net(x)
    print(f"z:{z.shape}, x_hat:{x_hat.shape}")


def test_FoldEncodedClassifier():
    hz = 64
    input_dim = hz*30
    net = FoldEncodedClassifier(
        n_block=5, input_dim=input_dim, kernel_sz=11, n_class=2, input_split=4
    )
    print(net)
    x = torch.randn(32, 1, input_dim)
    out = net(x)
    print(f"out:{out.shape}")


def test_FoldAEClassifier():
    input_dim = 30*64
    net = FoldEncoderClassifier(
        input_dim=input_dim
    )
    print(net)
    x = torch.randn(32, 1, input_dim)
    out = net(x)
    print(f"out:{out.shape}")


if __name__ == "__main__":
    test_classicCNN()
    # test_EEGConv()
    # test_FoldAEClassifier()
    # test_tcn()