class Plugin:
    async def close(self):
        pass

    async def run(self):
        raise NotImplementedError
