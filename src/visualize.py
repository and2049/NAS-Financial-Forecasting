import os
import json
from graphviz import Digraph
from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def plot(genotype, filename):
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    steps = len(genotype) // 2
    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    # draw edges based on the genotype
    for i, (op, j) in enumerate(genotype):
        if j == 0:
            u = "c_{k-2}"
        elif j == 1:
            u = "c_{k-1}"
        else:
            u = str(j - 2)

        v = str(i // 2)
        g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')

    # edges from intermediate nodes to the output node
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # render the graph
    g.render(filename, view=False, cleanup=True)
    print(f"Architecture diagram saved to {filename}.png")


def main():
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

    plot(genotype.normal, os.path.join("reports", "normal_cell"))
    plot(genotype.reduce, os.path.join("reports", "reduction_cell"))


if __name__ == '__main__':
    main()
