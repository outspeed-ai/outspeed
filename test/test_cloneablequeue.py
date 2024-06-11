import asyncio

from realtime.utils import CloneableQueue


async def main():
    q1 = CloneableQueue()
    q2 = await q1.clone()
    q3 = await q1.clone()

    await q1.put("hello")
    print(await q2.get())  # Outputs 'hello'
    print(await q3.get())  # Outputs 'hello'

