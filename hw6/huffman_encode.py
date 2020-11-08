import argparse

import torch

from net.models import AlexNet
from net.huffmancoding import huffman_encode_model, huffman_decode_model
import util

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('--model', default='saves/108062566.ptmodel', type=str, help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--log', type=str, default='log.txt', help='log file name')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

# encode
print(f"--- Load model ({args.model}) ---")
model = torch.load(args.model)
print('--- Accuracy before encoding ---')
accuracy = util.test(model, use_cuda)
util.log(f"accuracy_before_encoding {accuracy}", args.log)
print(f"--- Start encoding ---")
huffman_encode_model(model)

# decode
print(f"--- Start decoding ---")
model = AlexNet().to(device)
huffman_decode_model(model)
print(f"--- Accuracy after decoding ---")
util.test(model, use_cuda)
accuracy = util.test(model, use_cuda)
util.log(f"accuracy_after_decoding {accuracy}", args.log)



