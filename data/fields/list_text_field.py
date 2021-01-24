# from typing import Dict
# import pdb
# from allennlp.data.fields import ListField
# from overrides import overrides
#
#
# class ListTextField(ListField):
#     @overrides
#     def get_padding_lengths(self) -> Dict[str, int]:
#         padding_lengths = super().get_padding_lengths()
#         padding_lengths['total_num_tokens'] = padding_lengths['num_fields'] * \
#             padding_lengths['list_num_tokens']
#         return padding_lengths      # {'num_fields': 23, 'list_num_tokens': 9, 'list_roberta_length': 9,
#         # 'list_roberta_copy_masks_length': 9, 'total_num_tokens': 207}
