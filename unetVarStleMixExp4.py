from abc import abstractmethod
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from inspect import isfunction
import math
import random
import scipy
from scipy.ndimage import label, center_of_mass
import logging

logging.basicConfig(
    #format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./logs/train.log'),  # Add a FileHandler
        logging.StreamHandler()  # Add a StreamHandler for console output
    ]
)
logger = logging.getLogger('')
#logger = logging.getLogger('wordStylistGenerationLogs2')
logger.info('--- wordStylistGenerationLogs2 ---')



def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        
        ctx.input_tensors = [x.float().detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding




# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if q.shape != k.shape:        
            simClone = sim.clone()

            #print("\n\t 3 simClone.shape = ", simClone.shape) 

            simClone = rearrange(simClone,'(b h) n d -> b h n d', h =h)
            
            #print("\n\t 3.1 simClone.shape = ", simClone.shape) 
            #logger.info("\n\t 3.1 simClone.shape = {}".format(simClone.shape))
            # 3.simClone.shape =  torch.Size([2, 4, 256, 10])
            # 	 3.1 simClone.shape =  torch.Size([128, 4, 256, 10])

            #simClone = simClone.sum(dim=1)
            #simClone= simClone.squeeze(1)

            BS,noHeads,htwidth , numChars = simClone.shape
            
            if htwidth==256:
                simClone = simClone.view(BS,noHeads, 8, 32, numChars)
            elif htwidth==64:
                simClone = simClone.view(BS,noHeads, 4, 16, numChars)

            #print("\n\t 3.2 simClone.shape = ", simClone.shape) 
            #logger.info("\n\t 3.2 simClone.shape = {}".format(simClone.shape))

            simClone[:,:,:,:,0] = torch.roll(simClone[:,:,:,:,0], shifts=-4, dims=3)  # shift vertically by 1
 
        
            simClone = simClone.view(BS,noHeads,htwidth,numChars)
            simClone = rearrange(simClone,'b h n d -> (b h) n d', h =h)
            #sim = simClone
            
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        #print("#########################")
        #logger.info(f"{attn[:,:,0].shape}, {v[:,2,:].shape}")
        #print(f"\n\t attn[:,:,0].shape = {attn[:,:,0].shape} v[:,2,:].shape = {v[:,2,:].shape} ")
        
        # 	 attn[:,:,0].shape = torch.Size([32, 256]) v[:,2,:].shape = torch.Size([32, 80]) 
    	#     out.shape = torch.Size([32, 256, 80]) attn.shape = torch.Size([32, 256, 10]) v.shape = torch.Size([32, 10, 80])

        if q.shape != k.shape:
            out = einsum('b i j, b j d -> b i d', attn, v)
        else:
            out = einsum('b i j, b j d -> b i d', attn, v)
        
        #logger.info(f"\n\t 1.out.shape = {out.shape} attn.shape = {attn.shape} v.shape = {v.shape}")
        #print("\n\t 1.out.shape = ", out.shape," attn.shape = ", attn.shape," v.shape = ", v.shape)
        
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        #logger.info(f"\n\t 2.out.shape = {out.shape} attn.shape = {attn.shape} v.shape = {v.shape}")
        #print("\n\t 2.out.shape = ", out.shape," attn.shape = ", attn.shape," v.shape = ", v.shape)
        
        #return self.to_out(out)
        
        """
            adding 1 extra dimension for the head
        """        
        
        attn = attn.unsqueeze(1)
        attn = rearrange(attn,'(b h) 1 n d -> b h n d', h =h)                
        
        return self.to_out(out), attn
        
def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    #'seq shape', seq.shape)
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the image
        self.attnc = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention for the context
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
    """
    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
    """
    def forward(self, x, context=None):
        # Pass the inputs through the _forward method with checkpointing
        x, attn2 = checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        return x,attn2

    
    """
    def _forward(self, x, context=None):
        
        dummy_attn = torch.ones((2, 320, 4, 8), device=x.device)
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=None) + x
        x = self.ff(self.norm3(x)) + x
        return x,dummy_attn
    """
    
    def _forward(self, x, context=None):
        
                        
        xOrig = x.clone()
        
        x,attn1 = self.attn1(self.norm1(x))  
        x +=xOrig
        #x = self.attn1(self.norm1(x)) + x

        xOrig = x.clone()        
        x,attn2 = self.attn2(self.norm2(x), context=context, mask=None)  
        x +=xOrig
    
        x = self.ff(self.norm3(x)) + x
        
        return x,attn2


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, part='encoder', vocab_size=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))
        self.part = part
    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        #print('x spatial trans in', x.shape)
        # note: if no context is given, cross-attention defaults to self-attention
        
        #dummy_attn = torch.ones((2, 320, 4, 8), device=x.device)

        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        if self.part != 'sca':
            x = rearrange(x, 'b c h w -> b (h w) c')
    
        for block in self.transformer_blocks:
            x,attn2 = block(x, context=context)
        if self.part != 'sca':
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            attn2 = rearrange(attn2,'batch head (height width) noChars -> batch head height width noChars', height=h, width=w, noChars=attn2.shape[-1])

        x = self.proj_out(x)
        return x + x_in,attn2



# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, context):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential1(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        
        
        dummy_attn = torch.ones((x.shape[0], self.shape[0], self.shape[1], self.shape[2]), device=x.device)

        
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
                
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
                
            else:
                x = layer(x)
        return x


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def __init__(self, *args, return_attn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_attn = return_attn  # Determines whether to return attention maps

    def forward(self, x, emb, context=None,extraDict= None):
        # Placeholder for attention maps if required
        dummy_attn_maps = [] if self.return_attn else None

        #dummy_attn = torch.ones((2, 320, 14, 88), device=x.device)
        
        #print("\n\t self.return_attn = ", self.return_attn)
        
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb,extraDict)

            elif isinstance(layer, SpatialTransformer):
                if self.return_attn:
                    x,dummy_attn = layer(x, context)  # Assuming SpatialTransformer supports return_attn
                    dummy_attn_maps.append(dummy_attn)
                
                else:
                    x,dummy_attn = layer(x, context)
            
                
            else:
                x = layer(x)

        if self.return_attn:
            return x, dummy_attn_maps  # Return both output and attention maps
        return x  # Return only output if attention maps are not required


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(#dims,
                 self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = nn.AvgPool2d(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        allWriterEmbDict =None,
        max_x_coords= None,
        max_y_coords = None,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.allWriterEmbDict = allWriterEmbDict
        #self.max_x_coords = max_x_coords
        #self.max_y_coords = max_y_coords
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def add_emb_to_h(self,h, emb_out, num_columns):
        """
        Add emb_out to the first `num_columns` of the width of `h`.

        Parameters:
            h (torch.Tensor): The tensor to update, shape [batch, channels, height, width].
            emb_out (torch.Tensor): The tensor to add, shape [batch, channels, 1, 1].
            num_columns (int): The number of columns to add `emb_out` to.
        """
        # Ensure num_columns does not exceed the width of h
        num_columns = min(num_columns, h.size(3))
        
        # Expand emb_out to match the height and `num_columns` width
        expanded_emb_out = emb_out.expand(-1, -1, h.size(2), num_columns)

        # Add emb_out to the specified portion of h
        h[:, :, :, :num_columns] = h[:, :, :, :num_columns] + expanded_emb_out
        
        return h

    def add_emb_to_h_character_based(self, h, emb_out,emb_out1, max_x_coords, char_index,extraDict):
        """
        Add `emb_out` to `h` based on the maximum x-coordinate for a given character index across the batch.

        Parameters:
            h (torch.Tensor): The tensor to update, shape [batch, channels, height, width].
            emb_out (torch.Tensor): The tensor to add, shape [batch, channels, 1, 1].
            max_x_coords (torch.Tensor): Tensor of maximum x-coordinates, shape [batch, characters].
            char_index (int): Index of the character for which to extract the x-coordinates.

        Returns:
            Updated `h` tensor.
        """
        # Extract the x-coordinates for the given character index
        selected_x_coords = max_x_coords[:, char_index].to(h.device)  # Shape: [100] (batch)
        
        # Ensure emb_out is expanded to match the height and entire width of h
        batch_size, channels, height, width = h.shape  # h.shape: [100, 320, 8, 32]
        expanded_emb_out = emb_out.expand(-1, -1, height, width).to(h.device)  # emb_out.shape: [100, 320, 1, 1] -> expanded_emb_out.shape: [100, 320, 8, 32]
        expanded_emb_out1 = emb_out1.expand(-1, -1, height, width).to(h.device)  # emb_out.shape: [100, 320, 1, 1] -> expanded_emb_out.shape: [100, 320, 8, 32]

        #print("\n\t selected_x_coords.shape:",selected_x_coords.shape," h.shape:",h.shape)
        #print("\n\t expanded_emb_out.shape =",expanded_emb_out.shape)
        #print("\n\t h.device:",h.device," expanded_emb_out.device:",expanded_emb_out.device)
        
        # Create a mask for each sample in the batch
        width_indices = torch.arange(width, device=h.device).view(1, 1, 1, width).to(h.device)  # width_indices.shape: [1, 1, 1, 32]
        mask = width_indices < selected_x_coords.view(batch_size, 1, 1, 1)  # selected_x_coords.shape: [100] -> mask.shape: [100, 1, 1, 32]
        mask1 = width_indices > selected_x_coords.view(batch_size, 1, 1, 1)
        #Shuffle expanded_emb_out along the batch dimension
        #shuffle_indices = torch.randperm(batch_size, device=h.device)  # Random permutation of batch indices
        shuffled_emb_out = expanded_emb_out1 #expanded_emb_out[shuffle_indices]  # Shuffled along the batch dimension

        #print("\n\t width_indices.shape =",width_indices.shape," mask.shape =",mask.shape)
        
        h = mask1* shuffled_emb_out + h + expanded_emb_out * mask  # h.shape: [100, 320, 8, 32], expanded_emb_out.shape: [100, 320, 8, 32], mask.shape: [100, 1, 1, 32]

        #h = mask1* 0 + h + 0 * mask  # h.shape: [100, 320, 8, 32], expanded_emb_out.shape: [100, 320, 8, 32], mask.shape: [100, 1, 1, 32]

        #h = torch.zeroes_like(h) #mask1* shuffled_emb_out + h + expanded_emb_out * mask  # h.shape: [100, 320, 8, 32], expanded_emb_out.shape: [100, 320, 8, 32], mask.shape: [100, 1, 1, 32]

        # Apply the mask and add emb_out only to the specified region
        #h = h + expanded_emb_out * mask  # h.shape: [100, 320, 8, 32], expanded_emb_out.shape: [100, 320, 8, 32], mask.shape: [100, 1, 1, 32]

        """
            extraDict["emb"] = emb   
            extraDict["allWriterEmbDict"] = self.allWriterEmbDict
            extraDict["embeddings_from_dict"] = embeddings_from_dict
        
        """
        
        
        return h  # Updated h.shape: [100, 320, 8, 32]


    """
    max_x_coords=self.max_x_coords,
    max_y_coords=self.max_y_coords,

    """

    def forward(self, x, emb,extraDict):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb,extraDict), self.parameters(), self.use_checkpoint
        )


    def _forward(self, x, emb,extraDict):
        
        
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            
        # if context is None:
        #     context= torch.zeros(emb.shape).to(emb.device)
        
        # emb = torch.cat([emb, context], dim=-1)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
            
        charIndx = extraDict["charIndx"]

        if charIndx>=0:
            emb_out1 = self.emb_layers(extraDict["emb1"]).type(h.dtype)
            while len(emb_out1.shape) < len(h.shape):
                emb_out1 = emb_out1[..., None]
            
         
            
            
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            
            #print("\n\t emb_out.shape = ", emb_out.shape, " h.shape = ", h.shape)
            # 	 emb_out.shape =  torch.Size([8, 320, 1, 1])  h.shape =  torch.Size([8, 320, 8, 32])

            #h = h + emb_out
            
            # Assuming h.shape = [8, 320, 8, 32] and emb_out.shape = [8, 320, 1, 1]

            # Expand emb_out to match the height and half-width of h
            expanded_emb_out = emb_out.expand(-1, -1, h.size(2), h.size(3) // 2)

            # Add emb_out to the first half of the width of h
            #h[:, :, :, :h.size(3) // 2] = h[:, :, :, :h.size(3) // 2] + expanded_emb_out
            
            max_x_coords = extraDict["max_x_coords"]
            #max_y_coords = extraDict["max_y_coords"]
            #shuffle_indices = extraDict["shuffledWrIndx"]
            """
            try:
                print("\n\t max_x_coords.shape:",max_x_coords.shape)
                print("\n\t max_y_coords.shape:",max_y_coords.shape)
            except Exception as e:
                pass    
            """
            #print("\n\t h.shape:",h.shape," emb_out.shape:",emb_out.shape)
            #h = self.add_emb_to_h(h, emb_out, 24) # add_emb_to_h_character_based

            if not max_x_coords is None and charIndx>=0:
                #h = self.add_emb_to_h_character_based(h, emb_out,emb_out1, max_x_coords,charIndx,extraDict)
                #h = self.add_emb_to_h(hand, emb_out, 24)
                h = h + emb_out

            else:
                h = h + emb_out
            
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


##################################################################################

    
class Word_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Word_Attention, self).__init__()
        self.linear_query = nn.Linear(input_size, hidden_size)
        self.linear_key = nn.Linear(input_size, hidden_size)
        self.linear_value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)
        
        # Calculate attention scores
        scores = query @ key.transpose(-2, -1)
        scores = self.softmax(scores)
        
        # Calculate weighted sum of the values
        word_embedding = scores @ value
        return word_embedding


class CharacterEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, max_seq_len):
        super(CharacterEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = Word_Attention(hidden_size, hidden_size)

        self.embedding_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.positional_encoding = self.get_positional_encoding()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        #Remove positional encoding for ablation study
        x += self.positional_encoding[:x.size(1), :].to(x.device)
        word_embedding = self.attention(x)
        return word_embedding
    
    def get_positional_encoding(self):
        positional_encoding = torch.zeros(self.max_seq_len, self.embedding_dim)
        for pos in range(self.max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** (i / self.embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / self.embedding_dim)))
        return positional_encoding







##################################################################################

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=True,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=768,                 # custom transformer support
        vocab_size=256,                  # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=False,
        args=None, 
        max_seq_len=20,
        #mix_rate=0.5
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.args = args

        self.max_x_coords = None
        self.max_y_coords = None
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.max_seq_len = max_seq_len
        
        self.word_emb = CharacterEncoder(vocab_size, context_dim, max_seq_len).to(args.device)
        
        self.allWriterEmbeddings = None
        self.allWriterEmbDict = dict()

        #==================== INPUT BLOCK ====================
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        allWriterEmbDict=self.allWriterEmbDict,
                        max_x_coords=self.max_x_coords,
                        max_y_coords=self.max_y_coords,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            allWriterEmbDict=self.allWriterEmbDict,
                            max_x_coords=self.max_x_coords,
                            max_y_coords=self.max_y_coords,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        #==================== MIDDLE BLOCK ====================
        if self.args.attentionMaps==0:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    allWriterEmbDict=self.allWriterEmbDict,
                    max_x_coords=self.max_x_coords,
                    max_y_coords=self.max_y_coords,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    allWriterEmbDict=self.allWriterEmbDict,
                    max_x_coords=self.max_x_coords,
                    max_y_coords=self.max_y_coords,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                return_attn=True  
            )
            self._feature_size += ch

        
        if self.args.attentionMaps==1:
            #print("\n\t new middle block")
            self.middle_block = nn.ModuleList([])
            
            layers= [   
                    ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    allWriterEmbDict=self.allWriterEmbDict,
                    max_x_coords=self.max_x_coords,
                    max_y_coords=self.max_y_coords,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,),
                    ]
            
            layers.append(    
                    SpatialTransformer(ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim),
                    )
            
            self.middle_block.append(TimestepEmbedSequential(*layers))

            self.middle_block.append(TimestepEmbedSequential(

                    ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    allWriterEmbDict=self.allWriterEmbDict,
                    max_x_coords=self.max_x_coords,
                    max_y_coords=self.max_y_coords,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,)
            ))

            self._feature_size += ch
        
        
        #==================== OUTPUT BLOCK ====================
        
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        allWriterEmbDict=self.allWriterEmbDict,
                        max_x_coords=self.max_x_coords,
                        max_y_coords=self.max_y_coords,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        ) if not use_spatial_transformer else SpatialTransformer(
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            allWriterEmbDict=self.allWriterEmbDict,
                            max_x_coords=self.max_x_coords,
                            max_y_coords=self.max_y_coords,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            nn.Conv2d(model_channels, n_embed, 1),
            nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )
        
        self.interpolation = args.interpolation
        
    
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)
  
    
    import torch

    def get_max_attention_coordinates(self,attn_tensor):
        """
        Finds the maximum X and Y coordinates of the highest activation 
        in the attention tensor for all batch images and characters.
        
        Parameters:
            attn_tensor (torch.Tensor): Attention tensor of shape [Batch Size, Height, Width, Characters]
            
        Returns:
            max_x_coordinates (torch.Tensor): Tensor of maximum X coordinates, shape [Batch Size, Characters]
            max_y_coordinates (torch.Tensor): Tensor of maximum Y coordinates, shape [Batch Size, Characters]
        """
        # Extract the dimensions dynamically
        batch_size, height, width, num_chars = attn_tensor.shape

        # Flatten the height and width dimensions
        # The tensor is reshaped to [Batch Size, Height*Width, Characters]
        flattened_tensor = attn_tensor.view(batch_size, -1, num_chars)

        # Find the flat index of the maximum activation
        # Shape: [Batch Size, Characters]
        flat_indices = torch.argmax(flattened_tensor, dim=1)

        # Convert flat indices to (x, y) coordinates
        # X-coordinate: flat_indices % Width
        # Y-coordinate: flat_indices // Width
        max_x_coordinates = flat_indices % width
        max_y_coordinates = flat_indices // width

        return max_x_coordinates, max_y_coordinates

   
   
    import torch
    import numpy as np
    import scipy
    from scipy.ndimage import label, center_of_mass

    def get_blob_centroids(self,attn_tensor, sigma_multiplier=2):
        """
        Extracts the centroids of the blobs with maximum attention in an attention tensor.

        Parameters:
            attn_tensor (torch.Tensor): Attention tensor of shape [Batch Size, Height, Width, Characters].
            sigma_multiplier (float): Threshold multiplier for defining regions of interest (default is 1).

        Returns:
            max_x_coords (torch.Tensor): Tensor of maximum X coordinates (centroids), shape [Batch Size, Characters].
            max_y_coords (torch.Tensor): Tensor of maximum Y coordinates (centroids), shape [Batch Size, Characters].
        """
        # Convert the attention tensor to a numpy array for processing
        attn_numpy = attn_tensor.cpu().detach().numpy()

        # Extract dimensions
        batch_size, height, width, num_chars = attn_numpy.shape

        # Prepare containers for results
        max_x_coords = torch.zeros((batch_size, num_chars), dtype=torch.int32)
        max_y_coords = torch.zeros((batch_size, num_chars), dtype=torch.int32)

        # Iterate over batch and characters
        for b in range(batch_size):
            for c in range(num_chars):
                curr_attn_map = attn_numpy[b, :, :, c]

                # Normalize the attention map
                curr_attn_map = (curr_attn_map - curr_attn_map.min()) / (curr_attn_map.max() - curr_attn_map.min())

                # Calculate mean and standard deviation
                mean_val = curr_attn_map.mean()
                std_val = curr_attn_map.std()

                # Define threshold
                threshold = mean_val + sigma_multiplier * 2*std_val

                # Mask regions above the threshold
                thresholded_map = curr_attn_map >= threshold

                # Detect blobs using connected components
                labeled_map, num_features = scipy.ndimage.label(thresholded_map)

                if num_features > 0:
                    # Find the blob containing the maximum attention
                    max_idx = np.argmax(curr_attn_map[thresholded_map])
                    max_blob_label = labeled_map[thresholded_map][max_idx]

                    # Extract the blob mask
                    blob_mask = labeled_map == max_blob_label

                    # Compute the centroid of the blob
                    blob_centroid = scipy.ndimage.center_of_mass(blob_mask)

                    # Assign centroid values to results
                    max_y_coords[b, c] = int(blob_centroid[0])
                    max_x_coords[b, c] = int(blob_centroid[1])
                else:
                    # Default to the highest activation if no blobs are found
                    max_idx_flat = np.argmax(curr_attn_map)
                    max_y_coords[b, c], max_x_coords[b, c] = divmod(max_idx_flat, width)

        return max_x_coords, max_y_coords



   
    
    def forward(self, x, original_images=None, timesteps=None, context=None, y=None,y1=None,Attnmap=None,charIndx=None, original_context=None, or_images=None, mix_rate=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #print('y', y.shape)

        #print("\n\t Attnmap.shape:",Attnmap.shape)
        extraDict = dict()

        extraDict["max_x_coords"] =  None
        extraDict["max_y_coords"] =  None
        #extraDict["writerChange"] = writerChange
        
        #logger.info(f"\n\t writerChange:{writerChange}")
        #print("\n\t writerChange:",writerChange)
        
        if not Attnmap is None and charIndx>=0:
            #print("\n\t ,Attnmap.shape = ",Attnmap.shape)
            
            #max_x_coords, max_y_coords = self.get_max_attention_coordinates(Attnmap)
            
            max_x_coords, max_y_coords =  self.get_blob_centroids(Attnmap)

            extraDict["max_x_coords"] =  max_x_coords
            extraDict["max_y_coords"] =  max_y_coords
            
            #print("Max X-Coordinates Shape:", max_x_coords.shape)
            #print("Max X-Coordinates:\n", max_x_coords)
            #print("Max Y-Coordinates Shape:", max_y_coords.shape)
            #print("Max Y-Coordinates:\n", max_y_coords)
                
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        
        
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
        
        
        if 0:#self.allWriterEmbeddings is None:
            """
                take writerid from 0 to 700 and then take the corresponding embedding
                using  self.label_emb() 
            """
            
            allWriterID = torch.arange(0, 700).to(y.device)
            self.allWriterEmbeddings = self.label_emb(allWriterID)
            print("\n\t self.allWriterEmbeddings.shape = ",self.allWriterEmbeddings.shape)
        
        #if you want to explore interpolation between 2 random styles you can go to the --interpolation argument in the train.py file
        if self.interpolation:
            if mix_rate is not None:
                print('interpolation')
                s1 = random.randint(0, 338)
                s2 = random.randint(0, 338)
                while s1 == s2:
                    s2 = random.randint(0, 338)
                y1 = torch.tensor([s1]).long().to(x.device)
                y2 = torch.tensor([s2]).long().to(x.device)
                y1 = self.label_emb(y1).to(x.device)
                y2 = self.label_emb(y2).to(x.device)
            
                y = (1-mix_rate)*y1 + mix_rate*y2
                
                y = y.to(x.device)
                emb = emb + y #self.label_emb(y)
            else:
                emb = emb + self.label_emb(y) 
        else:
            
            if 0:
                if len(self.allWriterEmbDict.keys())<18:
                    for writerID in y:
                        if writerID not in self.allWriterEmbDict:
                            self.allWriterEmbDict[writerID.item()] = self.label_emb(writerID)
                
                #print("\n\t len(self.allWriterEmbDict.keys()) = ",len(self.allWriterEmbDict.keys()))           
                emb = emb + self.label_emb(y)  
                
            # Ensure the dictionary has at least two entries
            
            wrIDList = []
            wrIDList1 = []

            #print("\n\t y:",y)
            #print("\n\t y1:",y1)

            if len(self.allWriterEmbDict.keys()) < 340:#18
                for writerID in y:
                    # Check if the writer ID is already in the dictionary
                    if writerID.item() not in self.allWriterEmbDict:
                        # Add the writer's embedding to the dictionary
                        #self.allWriterEmbDict[writerID.item()] = self.label_emb(writerID.unsqueeze(0))  # Pass writerID as a tensor
                        
                        # Ensure writerID is on the same device as the embedding layer
                        #writerID_device = self.label_emb  # Get the device of the embedding layer
                        writerID = writerID.to(self.args.device)  # Move writerID to the same device

                        # Perform the embedding lookup
                        self.allWriterEmbDict[writerID.item()] = self.label_emb(writerID.unsqueeze(0))
                        
                    
                    wrIDList.append(writerID.item())
                
                if charIndx>=0:              
                    for writerID in y1:
                        # Check if the writer ID is already in the dictionary
                        if writerID.item() not in self.allWriterEmbDict:
                            # Add the writer's embedding to the dictionary
                            writerID = writerID.to(self.args.device)
                            self.allWriterEmbDict[writerID.item()] = self.label_emb(writerID.unsqueeze(0))  # Pass writerID as a tensor
                        
                        wrIDList1.append(writerID.item())

                    

            #logger.info("\n\t self.allWriterEmbDict.keys():%d",len(list(self.allWriterEmbDict)))
            #logger.info("\n\t self.allWriterEmbDict.keys():%s",list(self.allWriterEmbDict.keys()))
            #logger.info("\n\t self.allWriterEmbDict.keys():%s",wrIDList)
            #logger.info("\n\t y length: %d", y.size(0))  # Log the number of elements in y
            #logger.info("\n\t y values: %s", y.tolist())  # Convert the tensor to a list and log the values
            #logger.info("\n\t y1 values: %s", y1.tolist())

            # Retrieve embeddings for all writer IDs in `y`
            embeddings_from_dict = torch.stack(
                [self.allWriterEmbDict[writerID.item()] for writerID in y], dim=0
            )

            embeddings_from_dict = embeddings_from_dict.squeeze(1)
            # Add the retrieved embeddings to `emb`

            if charIndx>=0:              
                embeddings_from_dict1 = torch.stack([self.allWriterEmbDict[writerID.item()] for writerID in y1], dim=0)
                embeddings_from_dict1 = embeddings_from_dict1.squeeze(1)

            
            #temp = self.label_emb(y)
            
            """
            
            temp.shape =  torch.Size([100, 1280]) 	 emb.shape =  torch.Size([100, 1280])
            embeddings_from_dict.shape: torch.Size([100, 1, 1280])  y.shape: torch.Size([100])  
            writerID.unsqueeze(0).shape: torch.Size([1])

            """
            
            #print("\n\t temp.shape = ",temp.shape,"\t emb.shape = ",emb.shape)
            #print("\n\t embeddings_from_dict.shape:",embeddings_from_dict.shape," y.shape:",y.shape)#," writerID.unsqueeze(0).shape:",writerID.unsqueeze(0).shape) 

            extraDict["original_emb"] = emb
            
            emb = emb + embeddings_from_dict
            
            extraDict["emb"] = emb   
            extraDict["allWriterEmbDict"] = self.allWriterEmbDict
            extraDict["embeddings_from_dict"] = embeddings_from_dict
            
            if charIndx>=0:              
                extraDict["embeddings_from_dict1"] = embeddings_from_dict1
                
                #extraDict["emb1"] = extraDict["original_emb"]+ embeddings_from_dict1
                
                extraDict["emb1"] = extraDict["original_emb"]+ embeddings_from_dict1[:extraDict["original_emb"].shape[0]]


            extraDict["charIndx"] = charIndx
            #shuffledIndices = torch.randperm(y.shape[0], device=y.device)
            #extraDict["shuffledWrIndx"] = shuffledWrIndx
            #shuffledWriters = y[shuffledIndices]
            
        if context is not None:
            #Word embedding
            context = self.word_emb(context)
        
        
        h = x.type(self.dtype)
        
        #INPUT BLOCKS
        for module in self.input_blocks:
            h = module(h, emb, context,extraDict)
            hs.append(h)
        
        #MIDDLE BLOCK
        
        if self.args.attentionMaps==0:
            h,attn2 = self.middle_block(h, emb, context,extraDict)

            """
            h = x
            all_dummy_attn_maps = []
            for module in self.middle_block:
                
                print("\n\t len(module) =",len(module))
                h, dummy_attn_maps = module(h, emb, context)
                all_dummy_attn_maps.extend(dummy_attn_maps)
            """
            #print("\n\t dummy_attn_maps:",len(all_dummy_attn_maps))
            
        
        #OUTPUT BLOCKS
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context,extraDict)
            
        h = h.type(x.dtype)
        
        attn1,attn3 = None,None
         
         
        if  1:

            #print("\n\t 1. Attention shape:",len(attn2))
            
            #print("\n\t 1. Attention shape:",attn2[0].shape)
            
            attn2 = attn2[0]
            
            #print("\n\t 12. before sum Attention shape:",attn2.shape)

            attn2 =  attn2.sum(dim =1)
            attn2Original = attn2.clone() 
            
            #print("\n\t 21. sum Attention shape:",attn2.shape)
            
            # 1. sum Attention shape: torch.Size([500, 8, 32, 10])
            
            attn2 = F.interpolate(attn2.permute(0,3,1,2),scale_factor = (16,16),mode = "nearest")
            attn2 = attn2.permute(0,2,3,1)
            #print("\n\t 22.  Attention reshape:",attn2.shape)

             
        if self.predict_codebook_ids:
            
            if self.args.attentionMaps==1:
                return self.id_predictor(h),attn1,attn2,attn3,attn2Original
            else:
                return self.id_predictor(h),attn1,attn2,attn3,attn2Original
                
            
        else:
            
            if self.args.attentionMaps==1:
                return self.out(h),attn1,attn2,attn3,attn2Original
            else:
                return self.out(h),attn1,attn2,attn3,attn2Original




