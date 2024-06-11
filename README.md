## Development Environment

To run the dev environment, do:

```
docker-compose run --service-ports --rm realtime-runner bash
```

If the container is not already running, you have to run on container start:

```
poetry install --only main
```
