import json
import matplotlib.pyplot as plt
from pyannote.core import Annotation, Segment
from pyannote.core.notebook import Notebook
import pickle

def load_pickle(path:str):
    with open(path, "rb") as file:
        diar = pickle.load(file)

    annotation = Annotation()
    
    for entry in diar[0]:
        start, end, speaker = entry.split()
        
        start = float(start)
        end = float(end)

        segment = Segment(start=start, end=end)
        annotation[segment] = speaker
        
    return annotation

def load_json(json_source):
    with open(json_source, 'r', encoding='utf-8') as file:
        return json.load(file)

def load_json_annote(json_data, idx):
    annotation = Annotation()
    for item in json_data:
        if item['id'] == idx: 
            annotations = item['annotations']
            for ann in annotations:
                results = ann['result']
                for result in results:
                    start = result['value']['start']
                    end = result['value']['end']
                    labels = result['value']['labels']
                    for label in labels:
                        annotation[Segment(start, end)] = label
            break

    return annotation

def load_rttm_annote(file_path):
    annotation = Annotation()
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            start = float(parts[3])
            duration = float(parts[4])
            end = start + duration
            speaker = parts[7]
            annotation[Segment(start, end)] = speaker
    return annotation

def plot_annotation(annotation):
    plt.figure(figsize=(15, 5))
    notebook = Notebook()
    notebook.plot_annotation(annotation)
    plt.show()