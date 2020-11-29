import os
import nltk
import spacy
import jsonlines as jsonl
import requests
from tqdm import tqdm

def process_definition(defn):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(defn)
    return ''.join([token.text_with_ws for token in doc if token.lemma_.strip()])

def lookup(np, top=3):
    BASE_URL = 'http://api.urbandictionary.com/v0/define?term={{{}}}'
    response = requests.get(BASE_URL.format(np))
    if response.status_code != 200:
        return None
    try:
        definitions = response.json()['list']
        if not definitions:
            return None
        definitions = sorted(definitions, key=lambda x: x['thumbs_up'], reverse=True)
        definitions = definitions[:top]
        definitions = [process_definition(definition['definition'].replace('[','').replace(']', ''))
                        for definition in definitions]
    except:
        # import pdb; pdb.set_trace()
        print("Exception occurred for NP: {}".format(np))
        return None
    return definitions

def is_valid(np):
    if len(np) == 1 and (np[0].pos_ == "PRON" or np[0].pos_ == "DET" or np.root.is_stop):
        return False
    return True

def process_noun_phrase(np):
    if len(np) > 1 and (np[0].pos_ == "PRON" or np[0].pos_ == "DET"):
        np = np[1:]
    return np
    
def get_np_chunk_definitions(filepath):
    data = []
    nlp = spacy.load("en_core_web_sm")
    with jsonl.open(filepath) as reader:
        for i,obj in enumerate(tqdm(reader)):
            doc = nlp(obj["text"])
            np_chunks = [process_noun_phrase(np).text for np in doc.noun_chunks if is_valid(np)]
            obj["np_chunks"] = {}
            empty_np_chunks = []
            for np_chunk in np_chunks:
                np_defns = lookup(np_chunk)
                if np_defns:
                    obj["np_chunks"][np_chunk] = np_defns
                else:
                    empty_np_chunks.append(np_chunk)
            obj["empty_np_chunks"] = empty_np_chunks
            data.append(obj)

    return data

if __name__ == "__main__":
    root = "/home/amalad/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations"
    file_names = ["dev_seen", "dev_unseen", "test_seen", "test_unseen", "train"]
    for file_name in file_names:
        data_np_chunks = get_np_chunk_definitions(os.path.join(root, '{}.jsonl'.format(file_name)))
        with jsonl.open(os.path.join(root, '{}_ub_def.jsonl'.format(file_name)), mode='w') as writer:
            writer.write_all(data_np_chunks)