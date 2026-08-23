import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention1D(nn.Module):
 def __init__(self,c,reduction=8):
  super().__init__();h=max(8,c//reduction);self.net=nn.Sequential(nn.Linear(c,h),nn.GELU(),nn.Linear(h,c),nn.Sigmoid())
 def forward(self,x):return x*self.net(x.mean(-1)).unsqueeze(-1)

class MultiScaleBeatEncoder(nn.Module):
 """Lightweight raw 12-lead beat encoder; logits follow AAMI N/S/V/F."""
 def __init__(self,feature_dim=64,classes=4):
  super().__init__();self.stem=nn.Sequential(nn.Conv1d(12,48,7,stride=2,padding=3,bias=False),nn.BatchNorm1d(48),nn.GELU())
  self.branches=nn.ModuleList([nn.Conv1d(48,32,k,padding=k//2,bias=False) for k in (3,7,15)])
  self.mix=nn.Sequential(nn.BatchNorm1d(96),nn.GELU(),ChannelAttention1D(96),nn.Conv1d(96,feature_dim,5,stride=2,padding=2,bias=False),nn.BatchNorm1d(feature_dim),nn.GELU())
  self.head=nn.Linear(feature_dim,classes)
 def forward(self,x):
  h=self.stem(x.float());h=self.mix(torch.cat([b(h) for b in self.branches],1));feature=h.mean(-1)
  return {'feature':feature,'logits':self.head(feature)}

class TCNBackbone(nn.Module):
 def __init__(self,d=128,drop=.15):
  super().__init__();layers=[]
  for dilation in (1,2,4,8):layers += [nn.Conv1d(d,d,3,padding=dilation,dilation=dilation),nn.GroupNorm(8,d),nn.GELU(),nn.Dropout(drop)]
  self.net=nn.Sequential(*layers)
 def forward(self,x,mask):return (x+self.net(x.transpose(1,2)).transpose(1,2))*mask.unsqueeze(-1)
class GRUBackbone(nn.Module):
 def __init__(self,d=128,drop=.15):super().__init__();self.gru=nn.GRU(d,64,2,batch_first=True,bidirectional=True,dropout=drop);self.norm=nn.LayerNorm(d)
 def forward(self,x,mask):return self.norm(x+self.gru(x)[0])*mask.unsqueeze(-1)
class TransformerBackbone(nn.Module):
 def __init__(self,d=128,drop=.15):
  super().__init__();layer=nn.TransformerEncoderLayer(d,8,d*2,drop,batch_first=True,norm_first=True,activation='gelu');self.net=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
 def forward(self,x,mask):return self.net(x,src_key_padding_mask=~mask)*mask.unsqueeze(-1)

class MILTemporalClassifier(nn.Module):
 """Beat-supervised window logits with multiple-instance recording pooling."""
 def __init__(self,kind='tcn',d=128,events=4,max_windows=59,aux_dim=0):
  super().__init__();self.kind=kind;self.position=nn.Parameter(torch.randn(1,max_windows,d)*.005)
  self.aux_dim=aux_dim;self.aux_fusion=nn.Sequential(nn.LayerNorm(aux_dim),nn.Linear(aux_dim,d),nn.GELU(),nn.Dropout(.10)) if aux_dim else None
  cls={'tcn':TCNBackbone,'gru':GRUBackbone,'transformer':TransformerBackbone}[kind];self.backbone=cls(d)
  self.window_head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,events));self.attention=nn.Linear(d,events)
 def forward(self,x,valid_mask=None,aux=None):
  if valid_mask is None:valid_mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
  fused=x.float()
  if self.aux_fusion is not None:
   if aux is None:raise ValueError('auxiliary beat features are required')
   fused=fused+self.aux_fusion(aux.float())
  h=self.backbone(fused+self.position[:,:x.shape[1]],valid_mask);window=self.window_head(h);att=self.attention(h).masked_fill(~valid_mask.unsqueeze(-1),-1e4)
  # Noisy-or-like MIL: smooth maximum plus learned event attention.
  lse=torch.logsumexp(window.masked_fill(~valid_mask.unsqueeze(-1),-1e4)/.5,dim=1)*.5
  weighted=(torch.softmax(att,dim=1)*window).sum(1);return {'window_logits':window,'recording_logits':.5*(lse+weighted),'context':h,'event_attention':torch.softmax(att,dim=1)}

class ResidualWhitenedRetrieval(nn.Module):
 def __init__(self,d=128,whiten_mean=None,whiten_matrix=None):
  super().__init__();self.register_buffer('whiten_mean',torch.zeros(d) if whiten_mean is None else whiten_mean.float());self.register_buffer('whiten_matrix',torch.eye(d) if whiten_matrix is None else whiten_matrix.float());self.delta=nn.Linear(d,d,bias=False);nn.init.zeros_(self.delta.weight)
 def forward(self,context,valid_mask=None):
  if valid_mask is None:valid_mask=torch.ones(context.shape[:2],dtype=torch.bool,device=context.device)
  m=valid_mask.unsqueeze(-1);mean=(context*m).sum(1)/m.sum(1).clamp_min(1);white=(mean-self.whiten_mean)@self.whiten_matrix;raw=white+self.delta(white);return {'recording_raw':raw,'recording_embedding':F.normalize(raw,dim=1)}
