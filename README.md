<!--BEGIN_BANNER_IMAGE-->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/.github/outspeed_dark.jpg">
  <source media="(prefers-color-scheme: light)" srcset="/.github/outspeed_light.jpg">
  <img style="width:100%;" alt="The Outspeed Logo and SDK repository." src="https://raw.githubusercontent.com/outspeed/outspeed/main/.github/banner_light.png">
</picture>

<!--END_BANNER_IMAGE-->

# Outspeed

Outspeed is a PyTorch-inspired SDK for building real-time AI applications on voice and video input. It offers:

- Low-latency processing of streaming audio and video
- Intuitive API familiar to PyTorch users
- Flexible integration of custom AI models
- Tools for data preprocessing and model deployment

Ideal for developing voice assistants, video analytics, and other real-time AI applications processing audio-visual data.


## Install

You can install `outspeed` SDK from pypi using

```
pip install outspeed
```

This would install the core `outspeed` package.
You can read [docs](http://docs.outspeed.ai) to get started.

### Usage

You can read the [docs](http://docs.outspeed.ai) to learn more about the SDK.

To deploy your realtime function on Outspeed's infra, you can use the `outspeed deploy` CLI.

```
# functions.py contains your realtime function code
outspeed deploy --api-key=<your-api-key> functions.py
```

[Contact us](mailto:contact@outspeed.ai) to get an API key and deploy.

Once deployed, you can use the playground in the examples repo to test the deployed code.

### Examples

All the examples are available in the `examples` folder.
To install the package so that all examples run, use:

```
pip install 'outspeed[plugins,torch]'
```

Or, if you're using poetry:

```
poetry add 'outspeed[plugins,torch]'
```

This will install all the additional libraries that are required for examples to work.