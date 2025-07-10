import json


def generate_vid_tokens(total_toks, token_start = "VIDVID", token_end = "Z"):
    toks_list = list(range(0, total_toks))
    toks_list = [str(t) for t in toks_list]

    vid_tkn_chr_mapping = {chr(ord("A") + i): str(i) for i in range(10)}
    # output: {'A': '0', 'B': '1', 'C': '2', 'D': '3', 'E': '4', 'F': '5', 'G': '6', 'H': '7', 'I': '8', 'J': '9'}
    chr_vid_tkn_mapping = {j:i for i,j in vid_tkn_chr_mapping.items()}

    def remap(old_name: str) -> str:
        return "".join(chr_vid_tkn_mapping.get(c, c) for c in old_name)
    
    toks_list = [f"{token_start}{remap(t)}{token_end}" for t in toks_list]

    return toks_list

def create_cosmos_codebook(chemeleon_tokenizer_dir, chemeleon_cosmos_tokenizer_dir, vid_start_idx, total_cosmos_toks):
    with open(chemeleon_tokenizer_dir, 'r') as f:
        tokenizer = json.load(f)
        # dict_keys(['version', 'truncation', 'padding', 'added_tokens', 'normalizer', 
        #            'pre_tokenizer', 'post_processor', 'decoder', 'model'])
    
    ### 1
    tokenizer['version'] = '2.0' # 1.0 -> 2.0
    ### 2
    # tokenizer['truncation']
    ### 3
    # tokenizer['padding']
    ### 4
    vid_toks_list = generate_vid_tokens(total_toks=total_cosmos_toks)
    new_added_tokens = [
        {
            "id": vid_start_idx + tk_idx,
            "content": vtl,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        } for tk_idx, vtl in enumerate(vid_toks_list)
    ]
    tokenizer['added_tokens'].append(new_added_tokens)
    ### 5
    # tokenizer['normalizer']
    ### 6
    tokenizer["pre_tokenizer"]["pretokenizers"].append(
        {
            "type": "Split",
            "pattern": {
            "Regex": "(VIDVID)((A|B|C|D|E|F|G|H|I|J){1,5})Z"
            },
            "behavior": "Isolated",
            "invert": False
        }
    )
    ### 7
    # tokenizer["post_processor"]
    ### 8
    # tokenizer["decoder"]
    ### 9
    # tokenizer["model"]
    # dict_keys(['type', 'dropout', 'unk_token', 'continuing_subword_prefix', 'end_of_word_suffix', 'fuse_unk', 'vocab', 'merges'])
    new_vocab = {vtl: vid_start_idx + tk_idx for tk_idx, vtl in enumerate(vid_toks_list)}
    tokenizer["model"]["vocab"].update(new_vocab)

    with open(chemeleon_cosmos_tokenizer_dir, 'w') as f:
        json.dump(tokenizer, f, indent=4, separators=(',', ': '))


if __name__=="__main__":
    chemeleon_tokenizer_dir = "ckpts/chameleon/tokenizer/text_tokenizer.json"
    chemeleon_cosmos_tokenizer_dir = "ckpts/cosmos/tokenizer/text_tokenizer.json"
    vid_start_idx=65536
    total_cosmos_toks=64000
    create_cosmos_codebook(
        chemeleon_tokenizer_dir,
        chemeleon_cosmos_tokenizer_dir,
        vid_start_idx,
        total_cosmos_toks
    )


