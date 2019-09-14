import pandas as pd


if __name__ == '__main__':

    labels_file = '../../data/hf_round1_arrythmia.txt'
    samples_file = '../../data/hf_round1_label.txt'
    output_file = '../../data/train.csv'

    with open(labels_file, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    labels = [a.strip() for a in labels]
    n_labels = len(labels)

    with open(samples_file, 'r', encoding='utf-8') as f:
        samples = f.readlines()

    data = []
    for sample in samples:
        sample_items = sample.strip().split('\t')
        sample_labels = sample_items[3:]
        label = [0] * n_labels

        for sample_label in sample_labels:
            label[labels.index(sample_label)] = 1

        sample_items = sample_items[:3] + label
        data.append(sample_items)

    columns = ['file', 'age', 'sex'] + labels
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(output_file, index=False)
