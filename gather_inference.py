import os
import json
import jsonlines as jsonl
import pandas as pd
import argparse
import numpy as np


def load_jsonl(path):
	with open(path, 'r') as json_file:
		json_list = list(json_file)
	anns = []
	for json_str in json_list:
		result = json.loads(json_str)
		anns.append(result)
	return {x['id']:x for x in anns}
	
def run(args):
	# df = pd.DataFrame(columns=['Image ID', 'Image Path', 'Text', 'GT Label', 'Score', 'Prediction'])
	df = pd.DataFrame()
	# outfile = open(args.output, 'w')
	# outfile.write('Image ID,Image Path,Text,GT Label,Score,Prediction\n')
	results = pd.read_csv(args.results, delimiter=',')
	annotations = load_jsonl(args.annotation)
	for idx, i in enumerate(results.iterrows()):
		img_id = str(int(i[1]['id']))
		prob = i[1]['proba']
		pred = i[1]['label']
		ann = annotations[int(img_id)]
		# to_write = "{},{},{},{},{},{}\n".format(img_id, ann['img'], ann['text'], ann['label'], prob, pred)
		# outfile.write(to_write)
		df = df.append([{'Image ID': img_id, 'Image Path': ann['img'], 'Text': ann['text'], 'GT Label': ann['label'], 'Score': prob, 'Prediction': pred}])
	df.to_csv(args.output)
	# print(annotations)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--annotation', dest='annotation', required=True,  type=str)
	parser.add_argument('--results', dest='results', required=True,  type=str)
	parser.add_argument('--output', dest='output', required=True,  type=str)

	args = parser.parse_args()
	run(args)
