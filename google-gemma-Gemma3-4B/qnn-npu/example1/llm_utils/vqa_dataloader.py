import os
import re
import json
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class DatasetVQAv2(Dataset):
    def __init__(self, images_dir, questions_file, annotations_file) -> None:
        super().__init__()
        
        with open(questions_file, 'r') as q_file:
            self.questions = json.load(q_file)["questions"]
            
        with open(annotations_file, 'r') as a_file:
            self.annotations = json.load(a_file)["annotations"]
            
        self.images_dir = images_dir
        
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        q_s = self.questions[index]
        a_s = self.annotations[index]
        image_path = f"{self.images_dir}/COCO_val2014_{q_s['image_id']:012d}.jpg"
        
        return image_path, q_s['question'], q_s["question_id"], a_s['multiple_choice_answer']
    

def add_chat_template(question):
    message = [
        {
            "role": "system",
            "content": "Answer the question strictly in one word."
        },
        {
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question}"}
            ]
        }
    ]
    
    return message

