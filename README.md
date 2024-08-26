## Install

You can install `realtime-client` SDK from pypi using

```
pip install realtime-client
```

This would install the core `realtime-client` package.
You can read [docs](http://docs.getadapt.ai) to get started.

### Usage

You can read the [docs](http://docs.getadapt.ai) to learn more about the SDK.

To deploy your realtime function on Adapt's infra, you can use the `realtime deploy` CLI.

```
# functions.py contains your realtime function code
realtime deploy --api-key=<your-api-key> functions.py
```

[Contact us](mailto:contact@getadapt.ai) to get an API key and deploy.

Once deployed, you can use the playground in the examples repo to test the deployed code.

### Examples

All the examples are available at https://github.com/xAlpha8/realtime-examples repo.
To install the package so that all examples run, use

```
pip install "realtime-client[plugins]"
```

This will install all the additional libraries that are required for examples to work.

## Development Environment

To run the dev environment, do:

```
docker-compose run --service-ports --rm realtime-runner bash
```

If the container is not already running, you have to run on container start:

```
poetry install --only main
```

### Publishing

```
poetry config pypi-token.pypi your-api-token
```

```
poetry build
poetry publish
```
