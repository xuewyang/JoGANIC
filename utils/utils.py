import json
import os
import numpy as np
import pdb
import spacy
import cv2, random
# import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
# from detectron2.modeling import build_model
from PIL import Image
import numpy as np


who = ["PERSON"]
whohere = ["NORP", "ORG"]
when = ["TIME", "DATE"]
where = ["FAC", "GPE", "LOC"]
misc = ["PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE", "PERCENT", "MONEY", "QUANTITY",
        "ORDINAL", "CARDINAL", "EVENT"]
t_new = [{'who', 'where', 'when', 'misc', 'context'}, {'who', 'where', 'when', 'context'}, {'who', 'context'},
         {'who', 'when', 'context'}, {'who', 'when'}, {'who', 'where', 'when'}, {'who', 'where'}, {'where', 'context'},
         {'who', 'where', 'misc', 'context'}, {'who', 'when', 'misc', 'context'}, {'who', 'misc', 'context'},
         {'who', 'misc'}, {'who'}, {'who', 'where', 'context'}, {'context'}, {'who', 'where', 'misc'},
         {'where', 'when', 'context'}, {'who', 'when', 'misc'}, {'where', 'when', 'misc', 'context'},
         {'misc', 'context'}, {'where', 'misc', 'context'}, {'when', 'misc'}, {'when', 'misc', 'context'}, {'misc'},
         {'who', 'where', 'when', 'misc'}, {'when', 'context'}, {'where'},
         {'where', 'when', 'misc'}, {'where', 'when'}, {'where', 'misc'}, {'when'}, {''}]


nlp = spacy.load("en_core_web_lg")


def get_all_templates(folder, split):
    with open(os.path.join(folder, 'captions_' + split + '.json'), 'r') as f:
        caps = json.load(f)
    with open(os.path.join(folder, 'ners_' + split + '.json'), 'r') as f:
        ners_total = json.load(f)
    objects = np.load(os.path.join(folder, 'classes_' + split + '.npy'), allow_pickle=True)
    with open(os.path.join(folder, 'poss_' + split + '.json'), 'r') as f:
        poss = json.load(f)
    with open(os.path.join(folder, 'deps_' + split + '.json'), 'r') as f:
        deps = json.load(f)
    with open(os.path.join(folder, 'tags_' + split + '.json'), 'r') as f:
        tags = json.load(f)
    with open(os.path.join(folder, 'ners_' + split + '.json'), 'r') as f:
        ners_total = json.load(f)

    templates = {}
    templates_l = []
    for j, ners in enumerate(ners_total):
        template = []
        for i, ner in enumerate(ners):
            if ner in who:
                template.append('who')
            elif ner in when:
                template.append('when')
            elif ner in where:
                template.append('where')
            else:
                template.append('context')
        if 0 in objects[j]:
            template.append('who')
        if 'VERB' in poss[j]:
            template.append('context')
        # pdb.set_trace()
        template = set(template)
        if template in templates_l:
            template_t = templates_l[templates_l.index(template)]
        else:
            template_t = template
            templates_l.append(template_t)
        template_s = ' '.join(template_t)
        # if template_s == 'who':
        #     print(caps[j], ners)
        if template_s in templates.keys():
            templates[template_s] += 1
        else:
            templates[template_s] = 1
    print(templates)
    pdb.set_trace()


# get_all_templates(folder, 'test')


def get_objects(data_folder, split):
    image_dir = '/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes/images_processed'
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    with open(os.path.join(data_folder, 'images_' + split + '.json'), 'r') as j:
        img_ids = json.load(j)

    classes = []
    for i, image_id in enumerate(img_ids):
        image_path = os.path.join(image_dir, image_id + ".jpg")
        try:
            image = Image.open(image_path)
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            # image = cv2.imread(image_path)
        except (FileNotFoundError, OSError):
            continue
        outputs = predictor(image)
        objects = outputs['instances'].pred_classes
        objects = objects.cpu().numpy()
        classes.append(objects)

        if i % 2000 == 0:
            print(i)

    with open('/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes/valid_features/classes_valid.npy', 'wb') as f:
        np.save(f, classes)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# data_folder = '/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes/valid_features'
# split = 'valid'
# get_objects(data_folder, split)


def extract_caption_tree(folder, split):
    nlp = spacy.load("en_core_web_lg")
    ners = []
    ner_ts = []
    poss = []
    tags = []
    deps = []
    chunkss = []
    with open(os.path.join(folder, 'captions_reformated_' + split + '.json'), 'r') as f:
        caps = json.load(f)
    for i, caption in enumerate(caps):
        caption_doc = nlp(caption)
        ner = []
        ner_t = []
        pos = []
        tag = []
        dep = []
        chunks = []
        for ent in caption_doc.ents:
            label = ent.label_
            ner.append(label)
            ner_t.append(ent.text)
        for token in caption_doc:
            pos.append(token.pos_)
            tag.append(token.tag_)

        for chunk in caption_doc.noun_chunks:
            chunks.append(chunk.text)
            dep.append(chunk.root.dep_)
        ners.append(ner)
        ner_ts.append(ner_t)
        poss.append(pos)
        tags.append(tag)
        deps.append(dep)
        chunkss.append(chunks)
        if i % 2000 == 0:
            print(i)
    with open(os.path.join(folder, 'ners_reformated_' + split + '.json'), 'w') as f:
        json.dump(ners, f)
    with open(os.path.join(folder, 'nerts_reformated_' + split + '.json'), 'w') as f:
        json.dump(ner_ts, f)
    with open(os.path.join(folder, 'poss_reformated_' + split + '.json'), 'w') as f:
        json.dump(poss, f)
    with open(os.path.join(folder, 'tags_reformated_' + split + '.json'), 'w') as f:
        json.dump(tags, f)
    with open(os.path.join(folder, 'deps_reformated_' + split + '.json'), 'w') as f:
        json.dump(deps, f)
    with open(os.path.join(folder, 'chunkss_reformated_' + split + '.json'), 'w') as f:
        json.dump(chunkss, f)


folder = '/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes/test_features'
# extract_caption_tree(folder, 'test')


def get_whohere(cap, net):
    if net == '’m':
        return None
    doc = nlp(cap)
    text = [x.text for x in doc]
    if 'at' in text or 'in' in text or 'At' in text or 'In' in text:
        net = nlp(net)[0].text
        if net == '’':
            net = '’m'
        try:
            idx = text.index(net)
        except:
            return None
        if idx > 2:
            temp_text = [doc[idx - 2].text, doc[idx - 1].text]
            if 'at' in temp_text or 'in' in temp_text or 'At' in temp_text or 'In' in temp_text:
                return 'where'
    return 'who'


def get_labels_new(folder, split):
    assert split in ['train', 'valid', 'test']
    with open(os.path.join(folder, 'ners_' + split + '.json'), 'r') as f:
        ners_total = json.load(f)
    with open(os.path.join(folder, 'classes_' + split + '.npy'), 'rb') as f:
        objects = np.load(f, allow_pickle=True)
    with open(os.path.join(folder, 'poss_' + split + '.json'), 'r') as f:
        poss = json.load(f)
    with open(os.path.join(folder, 'captions_' + split + '.json'), 'r') as f:
        caps = json.load(f)
    with open(os.path.join(folder, 'nerts_' + split + '.json'), 'r') as f:
        nerts = json.load(f)
    t_labels = []
    for j, ners in enumerate(ners_total):
        t_label = [0, 0, 0, 0, 0]
        template = []
        for i, ner in enumerate(ners):
            if ner in who:
                template.append('who')
            elif ner in when:
                template.append('when')
            elif ner in where:
                template.append('where')
            elif ner in whohere:
                ll = get_whohere(caps[j], nerts[j][i])
                if ll is None:
                    continue
                template.append(ll)
            else:
                template.append('misc')
        if 0 in objects[j]:
            template.append('who')
        if 'VERB' in poss[j]:
            template.append('context')

        if 'who' in template:
            t_label[0] = 1
        if 'where' in template:
            t_label[1] = 1
        if 'when' in template:
            t_label[2] = 1
        if 'misc' in template:
            t_label[3] = 1
        if 'context' in template:
            t_label[4] = 1
        if len(template) == 0:
            t_label = [1, 1, 1, 1, 1]
        t_labels.append(t_label)

        if j % 2000 == 0:
            print(j)
    np.save(os.path.join(folder, 'labels_bin_' + split + '.npy'), t_labels)


# folder = '/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes/train_features'
# get_labels_new(folder, 'train')


def get_template_precision_recall(generation_file):
    with open(generation_file) as f:
        count = 0
        for line in f:
            count += 1

    ge = np.zeros([count, 5])
    ca = np.zeros([count, 5])
    with open(generation_file) as f:
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
            for token in caption_doc:
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

            if count % 200 == 0:
                print(count)

    for i in range(5):
        ge_i = ge[:, i]
        ca_i = ca[:, i]
        ge_i_ind = np.where(ge_i == 1)
        ca_i_ind = np.where(ca_i == 1)
        tp_i = np.intersect1d(ge_i_ind[0], ca_i_ind[0])
        recall_i = len(tp_i) / len(ca_i_ind[0])
        precision_i = len(tp_i) / len(ge_i_ind[0])
        print(recall_i, precision_i)


# generation_file = '/home/xuewyang/Xuewen/Research/TGNC/generations/nytimes/ne/generations.jsonl'
generation_file = '/home/xuewyang/Xuewen/Research/TGNC/generations/goodnews/ms/generations.jsonl'
get_template_precision_recall(generation_file)


def get_qualitative(generation_file):
    with open(generation_file) as f:
        count = 0
        for line in f:
            obj = json.loads(line)
            caption = obj['caption']
            generation = obj['generation']
            pdb.set_trace()


generation_file = '/home/xuewyang/Xuewen/Research/TGNC/expt/nytimes/tell/generations.jsonl'
# get_qualitative(generation_file)


def filter_generation(generation_folder, file_name, new_file_name):
    new_objs = []
    generation_file = os.path.join(generation_folder, file_name)
    new_generation_file = os.path.join(generation_folder, new_file_name)
    with open(generation_file) as f:
        for line in f:
            obj = json.loads(line)
            caption = obj['caption']
            generation = obj['generation']
            context = obj['context']
            # web_url = obj['web_url']
            # image_path = obj['image_path']
            # new_obj = {'caption': caption, 'generation': generation, 'context': context,
            #            'web_url': web_url, 'image_path': image_path}
            new_obj = {'caption': caption, 'generation': generation, 'context': context}
            new_objs.append(new_obj)

    with open(new_generation_file, 'w') as f:
        json.dump(new_objs, fp=f, indent=4, ensure_ascii=False)


generation_folder = '/home/xuewyang/Xuewen/Research/TGNC/expt/nytimes/fh_a/serialization'
file_name = 'generationsbest_gt2.jsonl'
new_file_name = 'generations_new_gt2.jsonl'
# filter_generation(generation_folder, file_name, new_file_name)
