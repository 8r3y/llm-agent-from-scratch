import asyncio
import time


async def tick():
    for i in range(5):
        print(f"  tick {i}")
        await asyncio.sleep(0.5)


async def bad_tool():
    time.sleep(2)  # блокирующий вызов, как requests.get / pandas.read_excel
    print("  bad_tool done")


async def good_tool():
    await asyncio.to_thread(time.sleep, 2)
    print("  good_tool done")


async def main():
    print("--- bad")
    await asyncio.gather(tick(), bad_tool())
    print("--- good")
    await asyncio.gather(tick(), good_tool())


asyncio.run(main())
