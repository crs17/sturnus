from transformers import GPT2LMHeadModel, AutoModelForCausalLM


def fetch_gpt2_from_huggingface(model="gpt2"):
  # Fetch the official OpenAI weights via Hugging Face
  hf_model = GPT2LMHeadModel.from_pretrained(model)
  openai_state_dict = hf_model.state_dict()

  return openai_state_dict


def fetch_granite_from_huggingface(model="ibm-granite/granite-3.0-1b-a400m-base"):
    hf_model = AutoModelForCausalLM.from_pretrained(model)
    state_dict = hf_model.state_dict()
    return state_dict



# This function will update an instance of our GPT2 implementation with parameters
# obtained from OpenAI via Huggingface.
#
# There a couple of things to be aware of:
# - The linear layer weights in the original GPT2 model ([N_input, N_output]) are 
#   transformed relative to the pyTorch Linear module ([N_output, N_input]).
# - In the original GPT2 model parameters for queries, keys and values are fused in 
#   tensors of shape [768, 2304]. So we need to split them into 3 [768, 768] tensors.

def load_hf_gpt2_weights(gpt, hf_state_dict):
    n_layers = len(gpt.trf_blocks)

    # Token and position embeddings
    gpt.tok_emb.weight.data.copy_(hf_state_dict["transformer.wte.weight"])
    gpt.pos_emb.weight.data.copy_(hf_state_dict["transformer.wpe.weight"])
    # Final norm
    gpt.final_norm.scale.data.copy_(hf_state_dict["transformer.ln_f.weight"])
    gpt.final_norm.shift.data.copy_(hf_state_dict["transformer.ln_f.bias"])
    # Out head projection
    gpt.out_head.weight.data.copy_(hf_state_dict["lm_head.weight"])

    # Set parameters for each of the transformer blocks
    for i in range(n_layers):
        block = gpt.trf_blocks[i]
        p = f"transformer.h.{i}"

        # Layer norms
        block.ln1.scale.data.copy_(hf_state_dict[f"{p}.ln_1.weight"])
        block.ln1.shift.data.copy_(hf_state_dict[f"{p}.ln_1.bias"])
        block.ln2.scale.data.copy_(hf_state_dict[f"{p}.ln_2.weight"])
        block.ln2.shift.data.copy_(hf_state_dict[f"{p}.ln_2.bias"])

        # Split fused QKV: [768, 2304] -> three [768, 768], then transpose for nn.Linear
        c_attn_w = hf_state_dict[f"{p}.attn.c_attn.weight"]
        c_attn_b = hf_state_dict[f"{p}.attn.c_attn.bias"]
        d = c_attn_w.shape[0]
        q_w, k_w, v_w = c_attn_w.split(d, dim=1)
        q_b, k_b, v_b = c_attn_b.split(d)

        block.att.W_Q.weight.data.copy_(q_w.T)
        block.att.W_K.weight.data.copy_(k_w.T)
        block.att.W_V.weight.data.copy_(v_w.T)
        block.att.W_Q.bias.data.copy_(q_b)
        block.att.W_K.bias.data.copy_(k_b)
        block.att.W_V.bias.data.copy_(v_b)

        block.att.out_projection.weight.data.copy_(
            hf_state_dict[f"{p}.attn.c_proj.weight"].T
        )
        block.att.out_projection.bias.data.copy_(
            hf_state_dict[f"{p}.attn.c_proj.bias"]
        )

        # Feed forward networks. Layer indices are 0 and 2 as layer index 1 
        # points to the GELU activation
        block.ffn.layers[0].weight.data.copy_(
            hf_state_dict[f"{p}.mlp.c_fc.weight"].T
        )
        block.ffn.layers[0].bias.data.copy_(
            hf_state_dict[f"{p}.mlp.c_fc.bias"]
        )
        block.ffn.layers[2].weight.data.copy_(
            hf_state_dict[f"{p}.mlp.c_proj.weight"].T
        )
        block.ffn.layers[2].bias.data.copy_(
            hf_state_dict[f"{p}.mlp.c_proj.bias"]
        )




def load_hf_granite_weights(model, hf_state_dict):
    n_layers = len(model.trf_blocks)

    # model.embed_tokens.weight torch.Size([49152, 1024])
    # model.norm.weight torch.Size([1024])
    # lm_head.weight torch.Size([49152, 1024])

    # Token embeddings
    model.tok_emb.weight.data.copy_(hf_state_dict["model.embed_tokens.weight"])
    # Final norm
    model.final_norm.weight.data.copy_(hf_state_dict["model.norm.weight"])
    # Out head projection
    model.out_head.weight.data.copy_(hf_state_dict["lm_head.weight"])



    # Set parameters for each of the transformer blocks
    for i in range(n_layers):
        # model.layers.0.self_attn.q_proj.weight torch.Size([1024, 1024])
        # model.layers.0.self_attn.k_proj.weight torch.Size([512, 1024])
        # model.layers.0.self_attn.v_proj.weight torch.Size([512, 1024])
        # model.layers.0.self_attn.o_proj.weight torch.Size([1024, 1024])

        # model.layers.0.input_layernorm.weight torch.Size([1024])
        # model.layers.0.post_attention_layernorm.weight torch.Size([1024])
        # model.layers.0.block_sparse_moe.router.weight torch.Size([32, 1024])
        # model.layers.0.block_sparse_moe.experts.gate_up_proj torch.Size([32, 1024, 1024])
        # model.layers.0.block_sparse_moe.experts.down_proj torch.Size([32, 1024, 512])

        block = model.trf_blocks[i]
        p = f"model.layers.{i}"
        print(p)
        # RMS norms
        block.rmsn1.weight.data.copy_(hf_state_dict[f"{p}.input_layernorm.weight"])
        block.rmsn2.weight.data.copy_(hf_state_dict[f"{p}.post_attention_layernorm.weight"])

        block.att.W_Q.weight.data.copy_(hf_state_dict[f"{p}.self_attn.q_proj.weight"])
        block.att.W_K.weight.data.copy_(hf_state_dict[f"{p}.self_attn.k_proj.weight"])
        block.att.W_V.weight.data.copy_(hf_state_dict[f"{p}.self_attn.v_proj.weight"])

        block.att.out_projection.weight.data.copy_(
            hf_state_dict[f"{p}.self_attn.o_proj.weight"]
        )

        # Routing weights
        block.moe.routing_weights.weight.data.copy_(hf_state_dict[f"{p}.block_sparse_moe.router.weight"])
        
        for e, expert in enumerate(block.moe.experts):
            gate, up = hf_state_dict[f"{p}.block_sparse_moe.experts.gate_up_proj"][e,...].tensor_split(2)
            down = hf_state_dict[f"{p}.block_sparse_moe.experts.down_proj"][e,...]

            expert.up_W.weight.data.copy_(up)
            expert.gate_W.weight.data.copy_(gate)
            expert.down_W.weight.data.copy_(down)
