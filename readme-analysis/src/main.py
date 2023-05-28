#!/usr/bin/env python3

import logging

import pandas as pd

from lda import GensimLDA, SKLearnLDA
from preprocess import Preprocess
from word_cloud import Cloud

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)
LOGGER = logging.getLogger()


def main(args):
    readme_data = pd.read_csv(args.input_file)
    logging.debug(readme_data.head())

    vectorizer, processed_data = Preprocess().run(readme_data)

    if args.cloud:
        Cloud().run(vectorizer, processed_data)

    if args.hyperparams:
        lda = GensimLDA(args.topics, args.workers, input_file=args.input_file)

        lda.test_model_hyperparams(vectorizer, processed_data,
                                   args.min_alpha, args.max_alpha, args.alpha_step,
                                   args.min_eta, args.max_eta, args.eta_step,
                                   args.min_topics, args.max_topics, args.topics_step)
    elif args.gensim:
        GensimLDA(args.topics, args.workers,
                  alpha=args.alpha, eta=args.eta).run(vectorizer, processed_data)
    else:
        SKLearnLDA(args.topics, args.workers).run(vectorizer, processed_data)


if __name__ == '__main__':
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from multiprocessing import cpu_count

    parser = ArgumentParser(prog='lda.py',
                            description='Latent Dirichlet Analysis for Repositories',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-v', '--verbose', action='store_true',
                       help='increase output verbosity')
    group.add_argument('-d', '--debug', action='store_true',
                       help='enable debug logging')
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(),
                        help='number of workers to use')
    parser.add_argument('-c', '--cloud', action='store_true',
                        help='generate word cloud')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_lda = subparsers.add_parser('lda', help='run LDA')
    lda_group = parser_lda.add_mutually_exclusive_group(required=True)
    lda_group.add_argument('-gs', '--gensim', action='store_true',
                           help='use Gensim implementation')
    lda_group.add_argument('-sk', '--sklearn', action='store_true',
                           help='use sklearn implementation')
    parser_lda.add_argument('-t', '--topics', type=int, default=11,
                            help='number of topics to use')
    parser_lda.add_argument('-a', '--alpha', type=float, default=0.36,
                            help='alpha to use')
    parser_lda.add_argument('-e', '--eta', type=float, default=0.34,
                            help='eta to use')
    parser_lda.add_argument('-tw', '--top-words', type=float, default=20,
                            help='top words to get on NMF')
    parser_lda.set_defaults(hyperparams=False)  # HACK

    parser_hy = subparsers.add_parser('hy', help='run hyperparameter tuning on Gensim LDA')
    parser_hy.add_argument('-mt', '--min-topics', type=int, default=2,
                           help='minimum number of topics to test')
    parser_hy.add_argument('-xt', '--max-topics', type=int, default=11,
                           help='maximum number of topics to test')
    parser_hy.add_argument('-ts', '--topics-step', type=int, default=1,
                           help='step size for topic testing')
    parser_hy.add_argument('-ma', '--min-alpha', type=float, default=0.01,
                           help='minimum alpha to test')
    parser_hy.add_argument('-xa', '--max-alpha', type=float, default=1,
                           help='maximum alpha to test')
    parser_hy.add_argument('-as', '--alpha-step', type=float, default=0.3,
                           help='step size for alpha testing')
    parser_hy.add_argument('-me', '--min-eta', type=float, default=0.01,
                           help='minimum eta to test')
    parser_hy.add_argument('-xe', '--max-eta', type=float, default=1,
                           help='maximum eta to test')
    parser_hy.add_argument('-es', '--eta-step', type=float, default=0.3,
                           help='step size for eta testing')
    parser_hy.set_defaults(hyperparams=True,
                           alpha=None, eta=None, topics=None)  # HACK

    parser.add_argument('input_file', help='input file')

    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.INFO)
    elif args.debug:
        LOGGER.setLevel(logging.DEBUG)

    main(args)
