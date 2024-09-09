import os

import click
import dill
import requests


@click.group()
def cli():
    """A CLI tool for deploying serialized Python files."""
    pass


@cli.command()  # This adds the deploy function as a sub-command of adapt
@click.argument(
    "file_path",
    type=click.Path(exists=True),
    required=True,
    metavar="FILE_PATH",
)
@click.option("--api-key", required=False, help="API key of the sending user")
@click.option("--base-url", required=False, help="Base URL of Adapt endpoint")
def deploy(file_path, api_key, base_url):
    """Serializes a .py file and sends it to the specified backend server."""
    BASE_URL = base_url or "https://infra.getadapt.ai"
    endpoint = f"{BASE_URL}/deploy"
    api_key = api_key or os.getenv("ADAPT_API_KEY")
    if not api_key:
        click.echo("No API key provided. Please set the ADAPT_API_KEY environment variable or use the --api-key flag.")
        return
    try:
        with open(file_path, "rb") as file:
            fmodule_content = file.read()

            payload = {"module": fmodule_content, "metadata": {}}
            serialized_payload = dill.dumps(payload, recurse=True)
            headers = {"Content-Type": "application/octet-stream", "X-API-KEY": api_key}
            response = None
            try:
                response = requests.post(endpoint, data=serialized_payload, headers=headers)
            except Exception as e:
                click.echo(f"An error occurred while connecting to the server: {str(e)}")
                click.echo(f"Base URL: {BASE_URL}, Headers: {headers}")
                if response:
                    click.echo(f"Response: {response.text}")
                return
            try:
                body = response.json()
            except Exception as e:
                click.echo(f"An error occurred while parsing the response: {str(e)}")
                click.echo(f"Response: {response.text}")
                return
            if response.status_code == 200:
                click.echo("Function successfully deployed!")
                click.echo(f"Use function URL: {BASE_URL}/run/{body['functionId']} to run the function")
            else:
                click.echo(f"Failed to deploy file. Status code: {response.status_code}")
                click.echo(f"Response: {response.text}")
    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    cli()
