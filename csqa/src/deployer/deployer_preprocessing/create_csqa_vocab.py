import operator
import pickle
from collections import Counter

import click

from utilities.corpus_preprocessing_utils.vocab_creation_utils import create_csqa_vocabs
from utilities.general_utils import create_sorted_dict


@click.command()
@click.option('-input_direc', help='path to input directory', required=True)
@click.option('-out', help='output path', required=True)
@click.option('-max_ctx_vocab_size', help='maximum vocabulary size of context utterances', required=True)
@click.option('-max_response_vocab_size', help='maximum vocabulary size of response utterances', required=True)
def main(input_direc, out, max_ctx_vocab_size, max_response_vocab_size):
    max_ctx_vocab_size = int(max_ctx_vocab_size)
    max_response_vocab_size = int(max_response_vocab_size)
    tuples = create_csqa_vocabs(input_direc)
    merged_ctx_dict = dict()
    merged_response_dict = dict()

    for tuple in tuples:
        sub_ctx_dict, sub_response_dict = tuple
        merged_ctx_dict = Counter(merged_ctx_dict) + Counter(sub_ctx_dict)
        merged_response_dict = Counter(merged_response_dict) + Counter(sub_response_dict)

    sorted_ctx_list = sorted(merged_ctx_dict.items(), key=operator.itemgetter(1))
    ctx_vocab_size = len(sorted_ctx_list)

    sorted_response_list = sorted(merged_response_dict.items(), key=operator.itemgetter(1))
    response_vocab_size = len(sorted_ctx_list)

    if ctx_vocab_size > max_ctx_vocab_size:
        sorted_ctx_list = sorted_ctx_list[:max_ctx_vocab_size]
    else:
        max_ctx_vocab_size = ctx_vocab_size

    if response_vocab_size > max_response_vocab_size:
        sorted_response_list = sorted_response_list[:max_response_vocab_size]
    else:
        max_response_vocab_size = response_vocab_size

    # Create dict
    sorted_ctx_dict = create_sorted_dict(sorted_list=sorted_ctx_list)
    sorted_response_dict = create_sorted_dict(sorted_list=sorted_response_list)

    parts_of_path = out.split('.pkl')
    out = parts_of_path[0] + '_context_vocab_' + str(max_ctx_vocab_size) + '.pkl'
    pickle.dump(sorted_ctx_dict, open(out, "wb"))

    out = parts_of_path[0] + '_response_vocab_' + str(max_response_vocab_size) + '.pkl'
    pickle.dump(sorted_response_dict, open(out, "wb"))


if __name__ == '__main__':
    main()
