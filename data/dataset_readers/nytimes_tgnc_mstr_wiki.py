import logging
import os
import random
from typing import Dict
import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from torchvision.transforms import (CenterCrop, Compose, Normalize, Resize, ToTensor)
from tqdm import tqdm
import json
import os
import pdb
from TGNC.data.fields import ImageField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register('nytimes_tgnc_mstr_wiki')
class NYTNewsTGNCMSTRWikiReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 image_dir: str,
                 base_folder: str,
                 eval_limit: int = 5120,
                 lazy: bool = True) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.image_dir = image_dir
        self.preprocess = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.eval_limit = eval_limit
        random.seed(1234)
        self.rs = np.random.RandomState(1234)
        self.base_folder = base_folder

    @overrides
    def _read(self, split: str):
        # split can be either train, valid, or test
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Unknown split: {split}')

        logger.info('Grabbing all article IDs')
        limit = self.eval_limit if split == 'valid' else 0
        data_folder = os.path.join(self.base_folder, split + '_features')
        with open(os.path.join(data_folder, 'images_wiki_' + split + '.json'), 'r') as j:
            img_ids = json.load(j)
        with open(os.path.join(data_folder, 'captions_wiki_' + split + '.json'), 'r') as j:
            captions = json.load(j)
        with open(os.path.join(data_folder, 'articles_wiki_' + split + '.json'), 'r') as j:
            articles = json.load(j)
        labels = np.load(os.path.join(data_folder, 'labels_wiki_bin_' + split + '.npy'), allow_pickle=True)
        ents = np.load(os.path.join(data_folder, 'ents_wiki_' + split + '.npy'), allow_pickle=True)
        ids = np.arange(len(img_ids))
        self.rs.shuffle(ids)
        if limit != 0:
            ids = ids[:limit]

        for sample_id in ids:
            # Find the corresponding article
            label = labels[sample_id]
            caption = captions[sample_id]
            article = articles[sample_id]
            ent = ents[sample_id][:70, :]
            # Load the image
            image_path = os.path.join(self.image_dir, img_ids[sample_id] + '.jpg')
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue

            yield self.article_to_instance(article, image, caption, image_path, label, ent)

    def article_to_instance(self, article, image, caption, image_path, label, ent) -> Instance:
        context = ' '.join(article.strip().split(' '))
        caption = caption.strip()

        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)

        if 'roberta_ms' in self._token_indexers.keys():
            token_indexer_cap = {'roberta': self._token_indexers['roberta']}
            token_indexer_art = {'roberta_ms': self._token_indexers['roberta_ms']}
        else:
            token_indexer_cap = {'roberta': self._token_indexers['roberta']}
            token_indexer_art = {'roberta': self._token_indexers['roberta']}

        fields = {
            'context': TextField(context_tokens, token_indexer_art),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, token_indexer_cap),
            'label': ArrayField(label, padding_value=np.nan),
            'ent': ArrayField(ent, padding_value=np.nan)
        }

        metadata = {'context': context,
                    'caption': caption,
                    'image_path': image_path}
        fields['metadata'] = MetadataField(metadata)

        return Instance(fields)
