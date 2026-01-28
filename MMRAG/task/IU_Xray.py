from datasets import load_dataset, load_from_disk
import os
from utils import compute_mdhash_id
from model_service.vlm_service import OpenAIVLMService
from torch.utils.data import Dataset

class IUXrayDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_path = f"./datasets/processed_datasets/{self.dataset_name}"
        if os.exist(self.dataset_path):
            self.dataset = load_from_disk(self.dataset_paths)
        else:
            self.dataset = self.process_dataset(self.dataset_name, len(self.dataset))
    
    def process_dataset(self, dataset_size=None):
        
        dataset = load_dataset(self.dataset_name, split="train") # load from network
        if dataset_size is not None:
            self.dataset = dataset.select(range(dataset_size))
        workspace_folder = self.dataset_path
        os.makedirs(os.path.join(workspace_folder, "images"), exist_ok=True)
        
        def form_content(ex):
            # 处理单个样本
            ex['content'] = ex['question'] + "<image>\nAnswer:" + ex['answer']
            
            ex['chunk_id'] = compute_mdhash_id(ex['content'], prefix="chunk_")
            image_paths = []
            for idx, image in enumerate(ex['images']):
                image_path = os.path.join(workspace_folder, "images", f"{ex['chunk_id']}_img{idx}.jpeg")
                image.save(image_path)
                image_paths.append(image_path)
            ex['image_paths'] = image_paths
            return ex
        
        dataset = dataset.map(form_content, num_proc=4)
        dataset.save_to_disk(self.dataset_path)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def evaluate(self, problem, completion):
        raise NotImplementedError
    