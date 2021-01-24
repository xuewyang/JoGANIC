import logging
import os
import random
import re
import pdb
import json
import cv2
import numpy as np
import spacy
import torch
from allennlp.common.util import prepare_environment
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.nn.util import move_to_device
from overrides import overrides
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize, ToTensor)
import sys
sys.path.append('..')
from TGNC.commands.train import yaml_to_params
from TGNC.data.fields import ImageField
from TGNC.models.resnet import resnet152


logger = logging.getLogger(__name__)
SPACE_NORMALIZER = re.compile(r"\s+")
ENV = os.environ.copy()


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class CaptioningWorker:
    def __init__(self, image_dir, base_folder, config_path, model_path, split='test'):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:0')
        else:
            self.device = torch.device('cpu')
        # loading configurations
        self.config_path = config_path
        logger.info(f'Loading config from {config_path}')
        config = yaml_to_params(config_path, overrides='')
        prepare_environment(config)
        vocab = Vocabulary.from_params(config.pop('vocabulary'))

        # loading model
        model = Model.from_params(vocab=vocab, params=config.pop('model'))
        self.model_path = model_path
        logger.info(f'Loading best model from {model_path}')
        best_model_state = torch.load(model_path, map_location=torch.device('cpu'))
        # best_model_state = torch.load(
        #     model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(best_model_state)
        model = model.eval()
        self.model = model.to(self.device)

        # loading roberta to get the dictionary
        logger.info('Loading roberta model.')
        roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
        self.bpe = roberta.bpe
        self.indices = roberta.task.source_dictionary.indices

        # loading resnet
        logger.info('Loading resnet model.')
        self.resnet = resnet152()
        self.resnet = self.resnet.to(self.device).eval()
        self.preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # loading data reader
        data_iterator = BasicIterator(batch_size=1)
        data_iterator.index_with(model.vocab)
        self.data_iterator = data_iterator

        # loading tokenizer, token indexers
        self.tokenizer = Tokenizer.from_params(config.get('dataset_reader').get('tokenizer'))
        indexer_params = config.get('dataset_reader').get('token_indexers')
        self.token_indexers = {k: TokenIndexer.from_params(p) for k, p in indexer_params.items()}
        self.split = split
        self.image_dir = image_dir

        # loading data
        self.base_folder = base_folder
        data_folder = os.path.join(self.base_folder, self.split + '_features')
        with open(os.path.join(data_folder, 'images_wiki_good_' + self.split + '.json'), 'r') as j:
            self.img_ids = json.load(j)
        with open(os.path.join(data_folder, 'articles_wiki_good_' + self.split + '.json'), 'r') as j:
            self.articles = json.load(j)
        with open(os.path.join(data_folder, 'captions_wiki_good_' + self.split + '.json'), 'r') as j:
            self.captions = json.load(j)
        self.labels = np.load(os.path.join(data_folder, 'labels_wiki_bin_good_' + self.split + '.npy'), allow_pickle=True)
        self.ents = np.load(os.path.join(data_folder, 'ents_wiki_good_' + self.split + '.npy'), allow_pickle=True)
        self.total = len(self.labels)

    def read_data(self, sample_id):
        # Find the corresponding article
        label = self.labels[sample_id]
        article = self.articles[sample_id]
        ent = self.ents[sample_id]
        # Load the image
        image_path = os.path.join(self.image_dir, self.img_ids[sample_id] + '.jpg')
        try:
            image = Image.open(image_path)
        except (FileNotFoundError, OSError):
            return

        return self.article_to_instance(article, image, image_path, label, ent)

    def fine_id(self, text):
        for i in range(len(self.labels)):
            caption = self.captions[i]
            if caption == text:
                return i
        return -1

    def article_to_instance(self, article, image, image_path, label, ent) -> Instance:
        context = ' '.join(article.strip().split(' '))
        context_tokens = self.tokenizer.tokenize(context)
        token_indexer_art = {'roberta': self.token_indexers['roberta']}
        # label = np.array([1, 1, 1, 0, 1])

        fields = {
            'context': TextField(context_tokens, token_indexer_art),
            'image': ImageField(image, self.preprocess),
            'label': ArrayField(label, padding_value=np.nan),
            'ent': ArrayField(ent, padding_value=np.nan)
        }

        metadata = {'context': context, 'image_path': image_path}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    def generate_one_caption(self, sample_id):
        instance = self.read_data(sample_id)
        instances = [instance]
        iterator = self.data_iterator(instances, num_epochs=1, shuffle=False)
        with torch.no_grad():
            for batch in iterator:
                if self.device.type == 'cuda':
                    batch = move_to_device(batch, self.device.index)
                output = self.model.generate(**batch)

        return output


def main():
    """
    This demonstrates how to use 2 different models to generate M captions for the same ids.
    Codes can be easily modified to use a different number of n models.
    """
    image_dir = '/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes/images_processed'
    base_folder = '/home/xuewyang/Xuewen/Research/data/Dataminr/nytimes'
    split = 'test'

    # the 1st model
    config_path_1 = '/home/xuewyang/Xuewen/Research/expt/nytimes/tell_wiki/config.yaml'
    model_path_1 = '/home/xuewyang/Xuewen/Research/expt/nytimes/tell_wiki/serialization/best.th'
    result_path_1 = '/home/xuewyang/Xuewen/Research/expt/nytimes/tell_wiki/serialization/samples_1.json'
    captioner_1 = CaptioningWorker(image_dir, base_folder, config_path_1, model_path_1, split)

    # the 2nd model
    config_path_2 = '/home/xuewyang/Xuewen/Research/expt/nytimes/tell_wiki/config.yaml'
    model_path_2 = '/home/xuewyang/Xuewen/Research/expt/nytimes/tell_wiki/serialization/best.th'
    result_path_2 = '/home/xuewyang/Xuewen/Research/expt/nytimes/tell_wiki/serialization/samples_2.json'
    captioner_2 = CaptioningWorker(image_dir, base_folder, config_path_2, model_path_2, split)
    # sample_id = 6211
    M = 10     # generate 10 captions for example
    random.seed(1234)  # seed the generator to get the same samples for each experiment
    ids = random.sample(range(0, 10000), M)  # generate num of ids used for caption generation

    to_save = []
    # start generation
    for id in ids:
        cap = captioner_1.generate_one_caption(id)
        to_save.append({str(id): cap})

    to_save = []
    for id in ids:
        cap = captioner_2.generate_one_caption(id)
        to_save.append({str(id): cap})

    with open(result_path_1, 'w') as f:
        json.dump(to_save, f, indent=4, ensure_ascii=False)

    with open(result_path_2, 'w') as f:
        json.dump(to_save, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()
