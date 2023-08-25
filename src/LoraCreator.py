from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class LORAFactory:
    def __init__(self, foundation_model_path, finetuned_model_path, rank=4):
        self.foundation_model_path = foundation_model_path
        self.finetuned_model_path = finetuned_model_path
        self.rank = rank
        # components to produce low-rank matrices, see LoRA paper, section 4.2 
        self.components = [
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj"
        ]

    def load_foundation_model(self):
        print(f"loading foundational model ...")
        self.foundation_model = AutoModelForSequenceClassification.from_pretrained(self.foundation_model_path)

    def load_finetuned_model(self):
        print(f"loading fine-tuned model ...") 
        self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.finetuned_model_path) 

    def load_models(self):
        self.load_foundation_model()
        self.load_finetuned_model()

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

    def create_lora(self):
        num_layers = len(self.foundation_model._modules["model"].layers)
        for layer_idx in range(num_layers):
            foundation_layer = self.foundation_model._modules["model"].layers[layer_idx]._modules["self_attn"]
            finetuned_layer = self.finetuned_model._modules["model"].layers[layer_idx]._modules["self_attn"]
            for comp in self.components:
                foundation_tensor = getattr(foundation_layer, comp).weight
                finetuned_tensor = getattr(finetuned_layer, comp).weight
                diff_tensor = finetuned_tensor - foundation_tensor
                low_rank_tensor = self.approximation(diff_tensor, self.rank)
                # override the finetuned model with the new tensor
                getattr(finetuned_layer, comp).weight = torch.nn.Parameter(low_rank_tensor)
        
    def save_lora(self, lora_model_path):
        if lora_model_path == self.finetuned_model_path:
            print(f"lora_model_path is the same as finetuned_model_path! Won't save to prevent overriding finetuned model")
            return
        self.finetuned_model.save_pretrained(lora_model_path)

    # Compute a rank-r approximation of given matrix
    def approximation(self, matrix, r):
        U, S, V = torch.linalg.svd(matrix)

        # Truncate the singular values and vectors
        U_r = U[:, :r]
        S_r = torch.diag(S[:r])
        V_r = V[:, :r]

        # Compute low-rank approximation
        return torch.matmul(U_r, torch.matmul(S_r, V_r.T))