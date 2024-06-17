import click
import dill
import requests
import os


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
@click.option("--endpoint", required=False, help="Endpoint to send the serialized file to.")
@click.option("--api-key", required=False, help="API key of the sending user")
def deploy(file_path, endpoint, api_key):
    """Serializes a .py file and sends it to the specified backend server."""
    BASE_URL = "https://infra.getadapt.ai"
    endpoint = os.getenv("ADAPT_ENDPOINT") or endpoint or f"{BASE_URL}/deploy"
    api_key = os.getenv("ADAPT_API_KEY") or api_key
    try:
        with open(file_path, "rb") as file:
            fmodule_content = file.read()

            payload = {"module": fmodule_content, "metadata": {}}
            serialized_payload = dill.dumps(payload, recurse=True)
            headers = {"Content-Type": "application/octet-stream", "X-API-KEY": api_key}
            response = requests.post(endpoint, data=serialized_payload, headers=headers)
            body = response.json()
            if response.status_code == 200:
                click.echo("Function successfully deployed!")
                click.echo(f"Use function URL: {BASE_URL}/run/{body['functionId']} to run the function")
            else:
                click.echo(f"Failed to deploy file. Status code: {response.status_code}")
    except Exception as e:
        click.echo(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    cli()
