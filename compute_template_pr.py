import numpy as np
import json
import pdb
import os


def compute_template_precision_recall(model_folder, generation_file):
    import spacy
    nlp = spacy.load("en_core_web_lg")
    who = ["PERSON"]
    whohere = ["NORP", "ORG"]
    when = ["TIME", "DATE"]
    where = ["FAC", "GPE", "LOC"]
    misc = ["PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "MONEY", "QUANTITY",
            "ORDINAL", "CARDINAL", "EVENT"]
    with open(os.path.join(model_folder, generation_file), 'r') as f:
        count = 0
        for line in f:
            count += 1

    ge = np.zeros([count, 5])
    ca = np.zeros([count, 5])
    with open(os.path.join(model_folder, generation_file), 'r') as f:
        count = 0
        for line in f:
            obj = json.loads(line)
            caption = obj['caption']
            generation = obj['generation']
            caption_doc = nlp(caption)
            generation_doc = nlp(generation)

            for ent in generation_doc.ents:
                label = ent.label_
                if label in who or label in whohere:
                    ge[count, 0] = 1
                elif label in when:
                    ge[count, 1] = 1
                elif label in where:
                    ge[count, 2] = 1
                elif label in misc:
                    ge[count, 3] = 1
            for token in generation_doc:
                if token.pos_ == 'VERB':
                    ge[count, 4] = 1
                    break

            for ent in caption_doc.ents:
                label = ent.label_
                if label in who or label in whohere:
                    ca[count, 0] = 1
                elif label in when:
                    ca[count, 1] = 1
                elif label in where:
                    ca[count, 2] = 1
                elif label in misc:
                    ca[count, 3] = 1
            for token in caption_doc:
                if token.pos_ == 'VERB':
                    ca[count, 4] = 1
                    break
            count += 1

    results = []
    prec_avg = []
    reca_avg = []
    for i in range(5):
        ge_i = ge[:, i]
        ca_i = ca[:, i]
        ge_i_ind = np.where(ge_i == 1)
        ca_i_ind = np.where(ca_i == 1)
        tp_i = np.intersect1d(ge_i_ind[0], ca_i_ind[0])
        recall_i = len(tp_i) / len(ca_i_ind[0])
        precision_i = len(tp_i) / len(ge_i_ind[0])
        print(precision_i, recall_i)
        if i == 0:
            results.append({"who_precision": precision_i, "who_recall": recall_i})
            prec_avg.append(precision_i)
            reca_avg.append(recall_i)
        if i == 1:
            results.append({"when_precision": precision_i, "when_recall": recall_i})
            prec_avg.append(precision_i)
            reca_avg.append(recall_i)
        if i == 2:
            results.append({"where_precision": precision_i, "where_recall": recall_i})
            prec_avg.append(precision_i)
            reca_avg.append(recall_i)
        if i == 3:
            results.append({"misc_precision": precision_i, "misc_recall": recall_i})
            prec_avg.append(precision_i)
            reca_avg.append(recall_i)
        if i == 4:
            results.append({"context_precision": precision_i, "context_recall": recall_i})
            prec_avg.append(precision_i)
            reca_avg.append(recall_i)
    print('avg precision: ', sum(prec_avg) / 5.0)
    print('avg recall: ', sum(reca_avg) / 5.0)
    with open(os.path.join(model_folder, "template_precision_recall.json"), 'w') as f:
        json.dump(results, f)


# model_folder = '/home/xuewyang/Xuewen/Research/TGNC/expt/goodnews/tell_no_image'
# generation_file = 'generations.jsonl'
# compute_template_precision_recall(model_folder, generation_file)

model_folder = '/home/xuewyang/Xuewen/Research/TGNC/expt/goodnews/tell'
generation_file = 'generations.jsonl'
compute_template_precision_recall(model_folder, generation_file)

model_folder = '/home/xuewyang/Xuewen/Research/TGNC/expt/goodnews/tell_full'
generation_file = 'generations.jsonl'
compute_template_precision_recall(model_folder, generation_file)

model_folder = '/home/xuewyang/Xuewen/Research/TGNC/expt/nytimes/tell_no_image'
generation_file = 'generations.jsonl'
compute_template_precision_recall(model_folder, generation_file)

model_folder = '/home/xuewyang/Xuewen/Research/TGNC/expt/nytimes/tell_roberta'
generation_file = 'generations.jsonl'
compute_template_precision_recall(model_folder, generation_file)

model_folder = '/home/xuewyang/Xuewen/Research/TGNC/expt/nytimes/tell_full'
generation_file = 'generations.jsonl'
compute_template_precision_recall(model_folder, generation_file)
