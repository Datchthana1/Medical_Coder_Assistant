import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class ICD10MatcherBERT:
    def __init__(self, icd_csv_path=None, model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 embeddings_path=None, load_saved=False):
        
        self.model = SentenceTransformer(model_name)
        
        if load_saved and embeddings_path:
            # โหลด embeddings ที่เซฟไว้ และข้อมูล ICD
            self.icd_embeddings = torch.load(embeddings_path)
            self.icd_data = pd.read_csv(icd_csv_path)
            self.icd_data['Medical_Coder'] = self.icd_data['Prefix_Header'] + "." + self.icd_data['Suffix_Header']
        elif icd_csv_path:
            self.icd_data = pd.read_csv(icd_csv_path)
            self.icd_data['Medical_Coder'] = self.icd_data['Prefix_Header'] + "." + self.icd_data['Suffix_Header']
            # สร้าง embeddings ใหม่
            self.icd_embeddings = self.model.encode(self.icd_data['Description_1'].tolist(), convert_to_tensor=True)
        else:
            raise ValueError("Either load_saved=True and embeddings_path must be given, or icd_csv_path must be provided.")
    
    def save_embeddings(self, save_path):
        torch.save(self.icd_embeddings, save_path)
    
    def predict(self, texts, top_n=5):
        text_embeddings = self.model.encode(texts, convert_to_tensor=True)
        results = []
        for idx, emb in enumerate(text_embeddings):
            cos_scores = util.cos_sim(emb, self.icd_embeddings)[0]
            top_results = cos_scores.topk(k=top_n)
            
            top_matches = []
            for score, i in zip(top_results.values, top_results.indices):
                i = i.item()
                top_matches.append({
                    'matched_code': self.icd_data.iloc[i]['Medical_Coder'],
                    'matched_desc': self.icd_data.iloc[i]['Description_1'],
                    'similarity': score.item()
                })
            
            results.append({
                'input': texts[idx],
                'matches': top_matches
            })
        return results


# ตัวอย่างใช้งาน

# สร้าง matcher และเซฟ embeddings ครั้งแรก
matcher = ICD10MatcherBERT(r'codes.csv')
matcher.save_embeddings(r'icd_embeddings.pt')
print('Embeddings and saved successfully!')

# ครั้งต่อไปโหลด embeddings จากไฟล์แทน encode ใหม่
matcher2 = ICD10MatcherBERT(
    icd_csv_path=r'codes.csv',
    load_saved=True,
    embeddings_path=r'icd_embeddings.pt'
)

input_texts = [

    "Cholera confirmed",
    "Typhoid fever, acute",
    "Paratyphoid fever B",
    "Pneumonia, bacterial",
    "Acute bronchitis",
    

    "Patient presents with severe dehydration due to cholera infection",
    "High fever and cough, diagnosed as typhoid pneumonia",
    "Symptoms indicate paratyphoid fever B with abdominal pain and headache",
    "Persistent cough and chest discomfort, suspect acute bronchitis",
    "Bacterial pneumonia with pleural effusion confirmed by x-ray",
    

    "Co-infection: typhoid fever and pneumonia, under treatment",
    "History of cholera, currently with acute bronchitis symptoms",
    

    "Fever with respiratory distress, unknown etiology",
    "Suspected gastrointestinal infection, awaiting lab results"
]


predictions = matcher2.predict(input_texts, top_n=5)

for pred in predictions:
    print(f"Input: {pred['input']}")
    for i, match in enumerate(pred['matches']):
        print(f" Match {i+1}: {match['matched_code']} - {match['matched_desc']} (Score: {match['similarity']:.4f})")
    print()
