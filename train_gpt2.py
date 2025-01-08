from dataclasses import dataclass
import torch.nn as nn
import torch
from torch.nn import functional as F




class CausalSelfAttention(nn.Module):
    """
    CausalSelfAttention performs multi-head self-attention in a causal (autoregressive) manner.
    This means tokens can only attend to their past positions.

    Dimensions to keep in mind:
      - B (batch size): The number of sequences in a single forward/backward pass.
      - T (block_size or sequence length): Maximum sequence length (e.g. 256).
      - C (n_embd): Embedding dimension (e.g. 384).
      - n_head: Number of attention heads (e.g. 6).
      - Each head has dimension: head_dim = n_embd / n_head (must be an integer).
    """

    def __init__(self, config):
        super().__init__()
        # For GPT, we typically have config.n_embd = 384, config.n_head = 6, etc.
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head for multi-head attention"

        # This linear layer will project the input embedding into Q, K, V
        # The projection is (n_embd) -> 3 * (n_embd) because it outputs Q, K, and V
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # This linear layer is used to combine the outputs of all heads back into a single embedding
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # (some additional attribute, presumably used for init scaling)

        # Save some config values for use in forward
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        """
        x has shape (B, T, C) where:
          - B: batch size,
          - T: sequence length (<= config.block_size),
          - C: embedding dimension (== config.n_embd).
        """
        B, T, C = x.size()  # B=batch size, T=sequence length, C=embedding dim (e.g. 384)

        # 1) Create Q, K, V by passing x through the linear layer:
        #    The result qkv has shape (B, T, 3*C).
        qkv = self.c_attn(x)
        # 2) Split the last dimension to separate Q, K, and V.
        #    Each now has shape (B, T, C).
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 3) Reshape each of Q, K, V so that the attention heads become a separate dimension.
        #    After view: (B, T, n_head, head_dim) => (B, T, n_head, C // n_head)
        #    Then transpose => (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_dim)

        # 4) Perform scaled dot-product attention (Flash Attention) in a causal manner,
        #    ensuring tokens only attend to previous tokens in the sequence.
        #    Output y will be (B, n_head, T, head_dim).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 5) Re-combine the n_head and head_dim dimensions back into a single embedding dimension.
        #    After transpose: (B, T, n_head, head_dim), then view => (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6) Final linear projection merges the heads back into a single embedding.
        #    Shape remains (B, T, C).
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

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

        x = x + self.attn(self.ln_1(x)) #comunnication between tokens
        x = x + self.mlp(self.ln_2(x)) #independent mapping of tokens

        return x        



@dataclass
class GPTConfig:
    block_size: int = 1024      # max seq length
    vocab_size: int = 50257     # tokens vocab: 50 000 BPE tokens + 256 bytes tokens + 1 <|endoftext|> special token
    n_layer: int = 12           # num of layers (head size)
    n_head: int = 12            # num of heads
    n_embd: int = 768           # embedding dimension



class GPT(nn.Module):

    def __init__(self, config : GPTConfig):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            dict(wte=nn.Embedding(config.vocab_size, config.n_embd), # embedding lookup for tokens
                 wpe=nn.Embedding(config.block_size, config.n_embd), # positional embeddings 
                 h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                 ln_f=nn.LayerNorm(config.n_embd)
                                  )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) #embedding -> next token

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):

        B, T = idx.size()

        assert T <= self.config.block_size, "Cannot forward sequence bigger than block size"

        #forward token and pos embeddings

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) #position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings of shape (B, T, n_embd)

        x = tok_emb + pos_emb 

        #forward blocks
        for block in self.transformer.h:

            x = block(x)

        #final layer
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1),), targets.view(-1))

        return logits, loss

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

        

import tiktoken

device = "cuda"

enc = tiktoken.get_encoding("gpt2")
with open("input.txt", "r") as f:
    text = f.read()

text =  text[:1000]
tokens = enc.encode(text)

B, T = 4, 32

buf = torch.tensor(tokens[: B*T + 1])
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
 
for i in range(50):

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()

    optimizer.step()

    print(f"#{i} LOSS: {loss}")
         


# #===========================GENERATE=====================================

# num_return_sequences = 3
# max_length = 50

# #model = GPT.from_pretrained("gpt2")
# model = GPT(GPTConfig())
# model.eval()


# model.to("cuda")


# #prefix tokens 

# import tiktoken

# enc = tiktoken.get_encoding("gpt2")

# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) #(8, )

# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(5, 8)

# x = tokens.to("cuda")

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)


# while x.size(1) < max_length:


#     with torch.no_grad():
#         logits = model(x)

#         logits = logits[:, -1, :] #(B, vocab_size)
#         probs = F.softmax(logits, dim=-1)

#         #sampling
#         topk_probs,  topk_indices  = torch.topk(probs, 50, dim=-1) #Sample only from top 50 most likely tokens

#         ix = torch.multinomial(topk_probs, 1)

#         #gather indices 
#         xcol = torch.gather(topk_indices, -1, ix) #col of new tokens

#         #append to sequence
#         x = torch.cat((x, xcol), dim=1)


# #print 

# for i in range(num_return_sequences):

#     tokens = x[i, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(">>>", decoded)
