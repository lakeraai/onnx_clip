"""
Open-CLIP Benchmarks
> https://github.com/mlfoundations/open_clip/blob/main/docs/openclip_results.csv

Open-CLIP Model Profile
> https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv

"""

HF_ViT_L_14 = {
    # 0.7921 % ImageNet-1k
    # https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
    "model_name": "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K",
    "embed_dim": 768,
}

EVA02_L_14_336 = {
    # 0.8039 % ImageNet-1k
    # https://github.com/baaivision/EVA/tree/master/EVA-CLIP
    "model_name": "EVA02-L-14-336",
    "tag": "merged2b_s6b_b61k",
    "embed_dim": 768,
}

HF_ViT_H_14_CLIPA_336 = {
    # 0.818 % ImageNet-1k
    # https://huggingface.co/UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B
    "model_name": "hf-hub/UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B",
    "embed_dim": 1024,
}

ViT_B_32 = {
    # 0.6332 % ImageNet-1k
    "model_name": "ViT-B-32",
    "tag": "openai",
    "embed_dim": 512,
}

RN50 = {
    # 0.5982 & ImageNet-1k
    "model_name": "RN50",
    "tag": "openai",
    "embed_dim": 1024,
}
