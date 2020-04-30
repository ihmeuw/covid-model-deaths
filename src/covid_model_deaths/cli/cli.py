from pathlib import Path

import click
from covid_shared import paths, cli_tools
from loguru import logger

from covid_model_deaths.cli import run_deaths


@click.command()
@cli_tools.pass_run_metadata()
@click.option('-s', '--input-version',
              type=click.Choice([paths.BEST_LINK, paths.LATEST_LINK]))
@click.option('-i', '--input-root',
              type=click.Path(file_okay=False),
              default=paths.MODEL_INPUTS_ROOT)
@click.option('-o', '--output-root',
              default=paths.DEATHS_OUTPUT_ROOT,
              show_default=True,
              type=click.Path(file_okay=False),
              help='Specify an output directory.')
@click.option('-b', '--mark-best', 'mark_dir_as_best',
              is_flag=True,
              help='Marks the new outputs as best in addition to marking them as latest.')
@click.option('-p', '--production-tag',
              type=click.STRING,
              help='Tags this run as a production run.')
@cli_tools.add_verbose_and_with_debugger
def deaths_model(run_metadata, input_version, input_root, output_root,
                 mark_best, production_tag, verbose, with_debugger):
    cli_tools.configure_logging_to_terminal(verbose)

    input_root = cli_tools.get_last_stage_directory(input_version, input_root, paths.MODEL_INPUTS_ROOT)
    run_metadata = cli_tools.update_with_previous_metadata(run_metadata, input_root)

    output_root = Path(output_root)
    cli_tools.setup_directory_structure(output_root.parent, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    run_metadata['output_path'] = str(run_directory)
    cli_tools.configure_logging_to_files(run_directory)

    main = cli_tools.monitor_application(run_deaths.run_deaths_model, logger, with_debugger)
    app_metadata, _ = main(input_root, output_root)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_best, production_tag)

    logger.info('**Done**')
