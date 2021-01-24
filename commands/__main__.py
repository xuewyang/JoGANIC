"""Train and run semantic diff models.

Usage:
    tell (train|evaluate) [options] PARAM_PATH
    tell (-h | --help)
    tell (-v | --version)

Options:
    -e --expt-dir EXPT_PATH
                        Directory to store experiment results and model files.
                        If not given, they will be stored in the same directory
                        as the parameter file.
    -r, --recover       Recover training from existing model.
    -f, --force    Delete existing models and logs.
    -o --overrides OVERRIDES
                        A JSON structure used to override the experiment
                        configuration.
    -u --pudb           Enable debug mode with pudb.
    -p --ptvsd PORT     Enable debug mode with ptvsd on a given port, for
                        example 5678.
    -g --file-friendly-logging
                        Outputs tqdm status on separate lines and slows tqdm
                        refresh rate
    -i --include-package PACKAGE
                        Additional packages to include.
    -q --quiet          Print less info
    -s --eval-suffix S  Evaluation generation file name [default: ]
    PARAM_PATH          Path to file describing the model parameters.
    -m --model-path PATH Path the the best model.

Examples:
    tell train -r -g expt/writing-prompts/lstm/config.yaml
"""

import logging
import os
import argparse
import ptvsd
import pudb
import pdb
from docopt import docopt
from schema import And, Or, Schema, Use
import sys
sys.path.append('..')
from TGNC.utils import setup_logger

from TGNC.commands.evaluate import evaluate_from_file
from TGNC.commands.train import train_model_from_file

logger = setup_logger()


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


parser = argparse.ArgumentParser(description='parse keys into a dictionary')
# Training setting.
parser.add_argument('--train', required=False, default=False, type=bool, help='Start training.')
parser.add_argument('--evaluate', required=False, default=False, type=bool, help='Start evaluation.')
parser.add_argument('--pudb', required=False, default=False, type=bool, help='pudb for debug.')
parser.add_argument('--recover', required=False, default=False, type=bool, help='Recover training from existing model.')
parser.add_argument('--eval_suffix', required=False, default='', type=str, help='.')
parser.add_argument('--model_path', required=False, default=None, type=str, help='.')
parser.add_argument('--quiet', required=False, default=False, type=bool, help='.')
parser.add_argument('--file_friendly_logging', required=False, default=False, type=bool, help='.')
parser.add_argument('--ptvsd', required=False, default=None, type=str, help='.')
parser.add_argument('--force', required=False, default=False, type=bool, help='.')
parser.add_argument('--include_package', required=False, default=None, type=str, help='.')
parser.add_argument('--overrides', required=False, default=None, type=bool, help='.')
parser.add_argument('--expt_dir', required=False, default=None, type=bool, help='.')
parser.add_argument('--param_path', required=True, default='', type=str, help='.')
parser.add_argument('--h', required=False, default=False, type=bool, help='.')
parser.add_argument('--v', required=False, default=False, type=bool, help='.')
parser.add_argument('--version', required=False, default=False, type=bool, help='.')
parser.add_argument('--debug', required=False, default=False, type=bool, help='.')


def validate(args):
    """Validate command line arguments."""
    args = {k.lstrip('-').lower().replace('-', '_'): v
            for k, v in args.items()}
    schema = Schema({
        'param_path': Or(None, os.path.exists),
        'model_path': Or(None, os.path.exists),
        'ptvsd': Or(None, And(Use(int), lambda port: 1 <= port <= 65535)),
        'eval_suffix': str,
        object: object,
    })
    args = schema.validate(args)
    args['debug'] = args['ptvsd'] or args['pudb']
    return args


def main(args):
    """Parse command line arguments and execute script."""
    # if args.debug:
    #     logger.setLevel(logging.DEBUG)
    if args.ptvsd:
        address = ('0.0.0.0', args.ptvsd)
        ptvsd.enable_attach(address)
        ptvsd.wait_for_attach()
    elif args.pudb:
        pudb.set_trace()

    if args.train:
        train_model_from_file(
            parameter_filename=args.param_path,
            serialization_dir=args.expt_dir,
            overrides=args.overrides,
            file_friendly_logging=args.file_friendly_logging,
            recover=args.recover,
            force=args.force)

    elif args.evaluate:
        evaluate_from_file(args.param_path, args.model_path,
                           args.overrides, args.eval_suffix)


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    main(args)
