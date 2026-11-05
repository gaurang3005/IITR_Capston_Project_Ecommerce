"""Live check against `uvicorn api:app --port 8000`: client closes mid-turn, then a new client."""
import asyncio, json, sys
import websockets

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8000/ws/chat"


async def main():
    async with websockets.connect(URL) as ws:                       # client 1: normal turn
        print("c1", json.loads(await ws.recv()))
        await ws.send(json.dumps({"message": "How much are 500 loyalty points worth?"}))
        print("c1 answer:", json.loads(await ws.recv())["outcome"])
        await ws.send(json.dumps({"message": "And what about Beauty?"}))
        # leave WITHOUT waiting for the reply: the server is mid-turn when the socket closes
    print("c1 disconnected mid-conversation")
    await asyncio.sleep(2)
    async with websockets.connect(URL) as ws:                       # client 2: server still up?
        await ws.recv()
        await ws.send(json.dumps({"message": "Check order ORD0029"}))
        print("c2 answer:", json.loads(await ws.recv())["outcome"])

asyncio.run(main())
