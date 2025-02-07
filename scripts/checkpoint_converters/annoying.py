from lightning.pytorch import Trainer

from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy


input_nemo_file = "/workspace/llama-3_1-8b-nemo_v1.0/llama3_1_8b.nemo"

dummy_trainer = Trainer(devices=1, accelerator='cpu', strategy=NLPDDPStrategy())