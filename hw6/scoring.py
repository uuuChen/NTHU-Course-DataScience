import torch
import util
import os
import csv
import numpy as np
import importlib
from net.ta_huffmancoding import ta_huffman_encode_model
from net.models import AlexNet


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
csv_file = './score_log.csv'
models_dir = './students'


def get_prune_score(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
    prune_rate = (total - nonzero) / total * 100.
    score = 0 if prune_rate == 0 else 70
    return prune_rate, score


def get_quantize_score(model):
    uni_elements_mean = 0
    num_of_layers = 0
    for name, p in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            tensor = p.data.cpu().numpy()
            uni_elements = np.unique(tensor)
            uni_elements_mean += len(uni_elements)
            num_of_layers += 1
    uni_elements_mean /= num_of_layers
    score = 0 if uni_elements_mean != 2 ** 5 + 1 else 10
    return uni_elements_mean, score


def person_test(model_path):
    score = 0
    csv_row = list()

    # get student id
    id = model_path.split('\\')[-1].split('.')[0]
    csv_row.append(id)
    module_obj = importlib.import_module(f'students.{id}')
    huffman_encode_model = getattr(module_obj, "huffman_encode_model")
    huffman_decode_model = getattr(module_obj, "huffman_decode_model")

    # load model
    try:
        model = torch.load(model_path)
        csv_row.append("S")
    except:
        csv_row.append("F")
        return csv_row, score

    # test prune
    prune_rate, prune_score = get_prune_score(model)
    csv_row.append(str(round(prune_rate, 2)))
    score += prune_score

    # test quantize
    cluster_nums_mean, quantize_score = get_quantize_score(model)
    csv_row.append(str(cluster_nums_mean))
    score += quantize_score

    # test model
    enc_acc = util.test(model, use_cuda)
    csv_row.append(f"{enc_acc}%")

    # get compression rate by our code
    try:
        compression_rate = ta_huffman_encode_model(model)
        if enc_acc >= 58:
            csv_row.append(str(round(compression_rate, 3)) + '*')
        else:
            csv_row.append("F")
    except:
        csv_row.append("F")

    # encode model by their code
    try:
        huffman_encode_model(model)
        csv_row.append("S")
        score += 5
    except:
        csv_row.append("F")
        return csv_row, score

    # decode model by their code
    model = AlexNet().to(device)
    try:
        huffman_decode_model(model)
        csv_row.append("S")
    except:
        csv_row.append("F")
        return csv_row, score

    # compare accuracies
    try:
        dec_acc = util.test(model, use_cuda)
        if enc_acc == dec_acc:
            score += 5
            csv_row.append("S")
        else:
            csv_row.append(f"F({enc_acc}, {dec_acc})")
    except:
        csv_row.append("F")

    return csv_row, score


def main():
    models_paths = [os.path.join(models_dir, file_name) for file_name in os.listdir(models_dir) if '.ptmodel' in file_name]
    compression_rankings = list()
    csv_rows = list()
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        fields = ['ID', 'Load Model', "Prune Rate", "Quantize clusters mean", "Before Encode Accuracy", 'Compression Rate', 'Encode Model', 'Decode Model', 'Accuracy Compare', 'Ranking', 'Score']
        writer.writerow(fields)
        for i, model_path in enumerate(models_paths):
            csv_row, score = person_test(model_path)
            csv_row += ["Nan"] * ((len(fields) - 2) - len(csv_row)) + [score]
            csv_rows.append(csv_row)
            id, compression_rate = csv_row[0], csv_row[5]
            if compression_rate != 'F':
                compression_rankings.append((id, compression_rate))
            if i == 1:
                break
        compression_rankings = sorted(compression_rankings, key=lambda x: float(x[1][:-1]), reverse=True)
        # one_third = int(len(models_paths) / 3)
        one_third = 1
        compression_rankings[:one_third] = [(id, 10) for (id, _) in compression_rankings[:one_third]]
        compression_rankings[one_third:one_third * 2] = [(id, 5) for (id, _) in compression_rankings[one_third:one_third * 2]]
        compression_rankings[one_third * 2:] = [(id, 0) for (id, _) in compression_rankings[one_third * 2:]]
        compression_rankings = dict(compression_rankings)
        for csv_row in csv_rows:
            id, score = csv_row[0], csv_row[-1]
            ranking_score = compression_rankings[id]
            score += ranking_score
            writer.writerow(csv_row[:-1] + [ranking_score, score])
        print(compression_rankings)


if __name__ == '__main__':
    main()
