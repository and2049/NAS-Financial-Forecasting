import os
import json
import torch
from torchviz import make_dot
from collections import namedtuple

import src.config as config
from src.model import Network


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def main():
    print("--- Loading genotype and building network ---")
    genotype_path = os.path.join('reports', 'genotype.json')
    if not os.path.exists(genotype_path):
        print(f"Genotype file not found at {genotype_path}. Please run search.py first.")
        return

    with open(genotype_path, 'r') as f:
        genotype_dict = json.load(f)

    genotype = Genotype(
        normal=genotype_dict['normal'],
        normal_concat=genotype_dict['normal_concat'],
        reduce=genotype_dict['reduce'],
        reduce_concat=genotype_dict['reduce_concat']
    )

    model = Network(
        C=config.INIT_CHANNELS,
        num_classes=2,
        layers=config.N_CELLS,
        genotype=genotype,
        dropout_p=config.DROPOUT_RATE
    )
    model.eval()


    dummy_input = torch.randn(1, config.INPUT_CHANNELS, config.SEQUENCE_LENGTH)

    print("--- Generating computational graph ---")
    y = model(dummy_input)

    output_filename = os.path.join("reports", "full_network_architecture")
    graph = make_dot(y, params=dict(model.named_parameters()))
    graph.render(output_filename, format='png', cleanup=True)

    print(f"\nFull network architecture diagram saved to {output_filename}.png")
    print(
        "Note: The graph can be very large and detailed. It's best to view it in an image viewer where you can zoom in.")


if __name__ == '__main__':
    main()
