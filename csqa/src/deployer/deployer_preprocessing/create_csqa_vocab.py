
import click
import pickle
from collections import Counter
from utilities.corpus_preprocessing_utils.vocab_creation_utils import create_csqa_vocabs
import operator

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

    sorted_ctx_dict = sorted(merged_ctx_dict.items(), key=operator.itemgetter(1))
    ctx_vocab_size = len(sorted_ctx_dict)

    sorted_response_dict = sorted(merged_response_dict.items(), key=operator.itemgetter(1))
    response_vocab_size = len(sorted_ctx_dict)

    if ctx_vocab_size > max_ctx_vocab_size:
        sorted_ctx_dict = sorted_ctx_dict[:max_ctx_vocab_size]
    else:
        max_ctx_vocab_size = ctx_vocab_size

    if response_vocab_size > max_response_vocab_size:
        sorted_response_dict = sorted_response_dict[:max_response_vocab_size]
    else:
        max_response_vocab_size = response_vocab_size

    parts_of_path = out.split('.pkl')
    out = parts_of_path[0] + '_context_vocab_' + str(max_ctx_vocab_size) + '.pkl'
    pickle.dump(sorted_ctx_dict, open(out, "wb"))

    out = parts_of_path[0] + '_response_vocab_' + str(max_response_vocab_size) + '.pkl'
    pickle.dump(sorted_response_dict, open(out, "wb"))

if __name__ == '__main__':
    main()