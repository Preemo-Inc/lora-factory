from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class LORAFactory:
    def __init__(self, foundation_model_path, finetuned_model_path, rank=2):
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
        self.load_models()

    def load_models(self):
        print(f"loading foundational model ...")
        self.foundation_model = AutoModelForSequenceClassification.from_pretrained(self.foundation_model_path)
        print(f"loading fine-tuned model ...") 
        self.finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.finetuned_model_path)

    def get_tensor_component(self, tensor_name):
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

    # Compute a rank-r approximation of given matrix
    def approximation(self, matrix, r):
        U, S, V = torch.linalg.svd(matrix)

        # Truncate the singular values and vectors
        U_r = U[:, :r]
        S_r = torch.diag(S[:r])
        V_r = V[:, :r]

        # Compute low-rank approximation
        return torch.matmul(U_r, torch.matmul(S_r, V_r.T))