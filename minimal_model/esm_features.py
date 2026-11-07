"""冻结 ESM-2 小模型的序列级特征提取。"""
from __future__ import annotations
import torch

class ESM2FeatureExtractor:
    def __init__(self, model_name='esm2_t6_8M_UR50D', device='auto', amp=True):
        import esm
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.device=torch.device(device); self.model,self.alphabet=getattr(esm.pretrained,model_name)()
        self.model=self.model.eval().to(self.device); self.batch_converter=self.alphabet.get_batch_converter()
        self.layer=self.model.num_layers; self.embedding_dim=int(self.model.embed_dim)
        self.amp=bool(amp and self.device.type=='cuda')
    @torch.no_grad()
    def encode(self,sequences,batch_size=64):
        out=[]
        for start in range(0,len(sequences),batch_size):
            current=[s.strip().upper() for s in sequences[start:start+batch_size]]
            _,strings,tokens=self.batch_converter([(str(i),s) for i,s in enumerate(current)])
            with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.amp):
                rep=self.model(tokens.to(self.device),repr_layers=[self.layer],return_contacts=False)['representations'][self.layer]
            out.extend(rep[i,1:len(s)+1].float().mean(0).cpu().numpy() for i,s in enumerate(strings))
        return out
    def attach(self,datasets,batch_size=64):
        samples=[s for ds in datasets for s in ds]; unique=list(dict.fromkeys(s['vh_sequence_aa'] for s in samples))
        mapping={s:e for s,e in zip(unique,self.encode(unique,batch_size))}
        for ds in datasets:
            for s in ds: s['esm_vh_embedding']=mapping[s['vh_sequence_aa']]
