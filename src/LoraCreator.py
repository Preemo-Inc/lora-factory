from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from fastchat.train.train_lora import get_peft_state_maybe_zero_3
from peft import LoraConfig, LoraModel, PeftModel, mapping
import torch
import pandas as pd
import json
import os

class LORAFactory:
    def __init__(self, foundation_model_path, finetuned_model_path, adapter_model_path, rank=4):
        if adapter_model_path==foundation_model_path or adapter_model_path==finetuned_model_path:
            print(f"ERROR: Please choose an adapter model path that is different from both the foundation model path and the finetuned model path.")
            return
        self.foundation_model_path = foundation_model_path
        self.finetuned_model_path = finetuned_model_path
        self.adapter_model_path = adapter_model_path
        self.rank = rank
        self.adapter_config = self.create_adapter_config()
        

    def load_foundation_model(self):
        print(f"load foundational model ...")
        # self.foundation_model = AutoModelForSequenceClassification.from_pretrained(self.foundation_model_path)
        self.foundation_model = AutoModelForCausalLM.from_pretrained(
            self.foundation_model_path 
        )

    def load_finetuned_model(self):
        print(f"load fine-tuned model ...") 
        # self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.finetuned_model_path) 
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            self.finetuned_model_path
        )

    def initialize_adapter(self):
        print("initialize adapter ...")
        adapter_config = LoraConfig.from_pretrained(self.adapter_model_path)
        self.adapter_model = mapping.get_peft_model(self.foundation_model, adapter_config)

    def load_models(self):
        self.load_foundation_model()
        self.load_finetuned_model()
        self.initialize_adapter()

    def get_module(self, model, module_name, lora=False):
        # module_name example: "model.layers.0.self_attn.k_proj"
        # they are recorded in the pytorch_model.bin.index.json file
        model = model._modules['model'] if not lora else model._modules['base_model'].model.model
        module_name = module_name.split(".")
        layer = int(module_name[2])
        layer_name = module_name[3]
        key = module_name[4] # k_proj, v_proj, etc.
        return  getattr(model.layers[layer]._modules[layer_name], key)
    
    def create_adapter_config(self):
        config = {
            "auto_mapping": None,
            "base_model_name_or_path": self.foundation_model_path,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "modules_to_save": None,
            "peft_type": "LORA",
            "r": self.rank,
            "revision": None,
            "target_modules": [
                "q_proj",
                "v_proj"
            ],
            "task_type": "CAUSAL_LM"
        }
        if not os.path.exists(self.adapter_model_path):
            os.makedirs(self.adapter_model_path)
        
        json_filename = os.path.join(self.adapter_model_path, "adapter_config.json")
        with open(json_filename, "w") as json_file:
            json.dump(config, json_file, indent=2)
        # pd.DataFrame(config).to_json(os.path.join(self.adapter_model_path, "adapter_config.json"))
        return config

    def get_tensor_component(self, tensor_name):
        # tensor_name example: "model.layers.0.self_attn.k_proj.weight"
        # they are recorded in the pytorch_model.bin.index.json file
        model = model._modules['model']
        tensor_name = tensor_name.split(".")
        assert len(tensor_name)==6 and tensor_name[1]=='layers' and tensor_name[5]=='weight'
        layer = int(tensor_name[2])
        layer_name = tensor_name[3]
        return getattr(model.layers[layer]._modules[layer_name], tensor_name[4])

    def get_tensor(self, model, tensor_name):
        component = self.get_tensor_component(tensor_name)
        return getattr(component, "weight")
    
    def write_tensor(self, model, tensor_name, tensor_value):
        component = self.get_tensor_component(tensor_name) 
        setattr(component, "weight", tensor_value)

    def create_tensor(self, tensor_name):
        foundation_tensor = self.get_tensor(self.foundation_model, tensor_name)
        finetuned_tensor = self.get_tensor(self.finetuned_model, layer_name)
        diff_tensor = finetuned_tensor - foundation_tensor
        low_rank_tensor = self.approximation(diff_tensor, self.rank)
        # override the finetuned model with the new tensor
        self.write_tensor(self.finetuned_model, tensor_name, low_rank_tensor)

    def create_adapter(self):
        target_modules = self.adapter_config["target_modules"]
        for name, finetuned_module in self.finetuned_model.named_modules():
            targeted = False
            for t in target_modules:
                if t in name:
                    targeted = True
                    break
            if targeted:
                if hasattr(finetuned_module, "weight"):
                    print(f"computing low rank matrix for {name} ...")
                    foundation_module = self.get_module(self.foundation_model, name)
                    adapter_module = self.get_module(self.adapter_model, name, lora=True)
                    delta = finetuned_module.weight - foundation_module.weight
                    A, B = self.approximation(delta, self.rank)
                    adapter_module.lora_A.default.weight = torch.nn.Parameter(A)
                    adapter_module.lora_B.default.weight = torch.nn.Parameter(B) 
        
        self.save_adapter()

    def save_adapter(self):
        state_dict = get_peft_state_maybe_zero_3(
            self.adapter_model.named_parameters(), self.adapter_config["bias"]
        )
        self.adapter_model.save_pretrained(self.adapter_model_path, state_dict=state_dict)
        # copy the tokenizer 
        tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path)
        tokenizer.save_pretrained(self.adapter_model_path)

    def merge_foundation_adapter_models(self, merged_model_path):
        assert merged_model_path not in [
            self.foundation_model_path,
            self.finetuned_model_path,
            self.adapter_model_path
        ]
        print(f"load foundational model ...")
        foundation_model = AutoModelForCausalLM.from_pretrained(
            self.foundation_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        merged_model = PeftModel.from_pretrained(
            foundation_model,
            self.adapter_model_path,
        )
        merged_model = merged_model.merge_and_unload()
        base_tokenizer = AutoTokenizer.from_pretrained(self.foundation_model_path, use_fast=False, legacy=False)
        merged_model.save_pretrained(merged_model_path)
        base_tokenizer.save_pretrained(merged_model_path)

    # Compute a rank-r approximation of given matrix.
    # See https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py#L752
    def approximation(self, matrix, r):
        U, S, Vh = torch.linalg.svd(matrix)

        # Truncate the singular values and vectors
        U = U[:, :r]
        S = S[:r]
        U = U @ torch.diag(S)
        Vh = Vh[:r, :]

        return Vh, U