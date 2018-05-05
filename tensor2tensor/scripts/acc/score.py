from __future__ import division

import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--decoded", dest="decoded", required=True)
    parser.add_argument("-r", "--reference", dest="reference", required=True)
    args = parser.parse_args()

    accuracy, precision, recall, atleast = metrics(args.decoded, args.reference)
    print("accuracy: {} precision: {} recall: {} atleast: {}".format(accuracy, precision, recall, atleast))

def metrics(decoded, reference):
    amount = 0
    atleast = 0
    right = 0
    total = 0
    precision = 0
    recall = 0
    with open(decoded) as decoded_file, open(reference) as reference_file:
        for decoded_line, reference_line in zip(decoded_file, reference_file):
            decoded_targets = decoded_line.strip().split(" ")
            reference_targets = reference_line.strip().split(" ")

            true_positives = score(decoded_targets, reference_targets)
            false_positives = len(decoded_targets) - true_positives
            false_negatives = len(reference_targets) - true_positives

            right += true_positives
            total += len(reference_targets)
            precision += true_positives / (true_positives + false_positives)
            recall += true_positives / (true_positives + false_negatives)
            if true_positives > 0:
                atleast += 1

            amount += 1
    return (right / total, precision / amount, recall / amount, atleast / amount)


def score(dec, ref):
    dec = list(set(dec))
    matches = 0;
    for a in dec:
        for b in ref:
            if (a == b):
                matches +=1
    return matches

main()
