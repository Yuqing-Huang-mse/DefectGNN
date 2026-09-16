"""Command-line entry point for DefectGNN training."""

from defectgnn import registers
from defectgnn.utils.common_util import CommonArgs, setup_imports


def main():
    setup_imports()
    cli = CommonArgs()
    cli.parser.set_defaults(config="train.yaml")
    parsed = cli.parser.parse_args()
    args = cli.get_args(parsed.config)
    task = registers.task.get_class(args.names.task_name)(args)
    task.run()


if __name__ == "__main__":
    main()

