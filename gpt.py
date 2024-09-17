import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

class DataLoaderV1:
    def __init__(self, B, T, dataset="shakespeare"):
        self.B = B
        self.T = T
        
        # Load enron dataset
        with open(self.get_datafile(dataset), 'r') as f:
            text = f.read()
        # Encode tokens for gpt2 with tiktoken
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens...")
        print(f"Loaded {len(self.tokens)} tokens...")
        self.current_posn = 0
        
    def get_datafile(self, dataset):
        if dataset == "shakespeare":
            return "datasets/tinyshakespeare.txt"
        elif dataset == "enron":
            return "datasets/enron_data.txt"
        else:
            raise Exception("Invalid Dataset!")
        
    def next_batch(self):
        B, T = self.B, self.T
        # Read the next tokens into the buffer
        # Recall the +1 is to get the last target token
        buf = self.tokens[self.current_posn : self.current_posn+(B*T+1)]
        # Take offsets for data and target
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        # Update posn, if it would be out of bounds, reset it 
        self.current_posn += B * T
        if self.current_posn + (B * T + 1) > len(self.tokens):
            self.current_posn = 0
        return x, y

class CausalSelfAttention(nn.Module):
    # Heads for MultiHead attention are like streams, their output is just concat
    # Each of the 1024 tokens emits a query, key, value
    # Vars named to match HuggingFace transformers code

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C) # This is a concatenation
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # self.gelu = nn.GELU(approximate='tanh')
        self.gelu = nn.GELU() # GPT2 uses the approximation, but we don't need to
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Think of this as Reduce (across tokens)
        x = x + self.mlp(self.ln_2(x)) # Think of this as Map (each token individually)
        return x
    
# Seems similar to a Java record, auto generates functions for typed attrs
@dataclass
class GPTConfig:
    block_size: int = 1024 # Max seq len
    vocab_size: int = 50257 # Number of tokens
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Layers have random initialization by default
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Weights of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Weights of position embeddings

            # Recall embedding is wrapper around tensor

            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)), # Hidden layer, index using integers
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # LM = language model
        
        # Share the weights between head and output, because we expect semantically similar words to have similar output probs
        # This saves us about 40 million params, we can be more efficient when training because of this intuition/bias
        self.transformer.wte.weight = self.lm_head.weight
        
        # Apply does this to every layer
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        # As a general rule, std should be 1/sqrt(size of model). This is roughly 0.02 (120 param = 1024)
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # In paper, positional embedding is 0.01. We use 0.02 here
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T < self.config.block_size, f"Cannot forward seq of len {T}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Make sure device (GPU) matches
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # Return a (B, T, vocab_size) tensor
        
        loss = None
        if targets is not None:
            # View is to flatten tensort to 2ds, B*T x vocab_size
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
        
    # FROM https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L214
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    
def get_enron_data():
    with open('enron_data.txt', 'r') as f:
        text = f.read()
        data = text[:1000] # First 1000 chars, ~300 tokens
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data)
        return tokens
        
    
def print_sample_data():
    tokens = get_enron_data()
    print(tokens[:24])
        

def get_data_batch(device):
    tokens = get_enron_data()
    B, T = 4, 32
    buf = torch.tensor(tokens[:B*T + 1])
    buf = buf.to(device)
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    return x, y
    
device = "cuda" if torch.cuda.is_available() else "cpu"  
print("Running training on device " + str(device))  
model = GPT(GPTConfig())
model.to(device)

# Using batch size 4x32
loader = DataLoaderV1(B=4, T=32, dataset="shakespeare")

# print_sample_data()
# x, y = get_data_batch(device)
# Note, when we calc and print loss from uninitialized, we expect each vocab to be roughly uniformly likely
# Taking the vocab size, using -ln(1/50000), we get ~11, which is what we observe
# logits, loss = model(x, y)
# print(loss)

# Adam and AdamW are alternatives to stoch grad desc
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    # Use our dataloader to get the next batch of data
    x, y = loader.next_batch()
    x, y = x.to(device), y.to(device)
    # Always remember to start with a 0 gradient
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # Loss is a tensor, item() converts it to a float
    # The tensor will live on the GPU, item moves the float to the CPU
    print(f"Step {i}, Loss {loss.item()}")