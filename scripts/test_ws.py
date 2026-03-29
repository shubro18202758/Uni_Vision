"""Quick test: connect WS first, then start processing — verify events arrive."""
import asyncio
import json
import requests
import websockets


async def listen():
    async with websockets.connect("ws://localhost:8000/ws/pipeline") as ws:
        print("WS connected — starting processing job...")

        # Start a processing job while we're connected
        r = requests.post("http://localhost:8000/pipeline/process", json={
            "source_url": r"C:\Users\sayan\Downloads\Uni_Vision\data\uploads\videos\6d6a9fd8154b_test1.mp4",
            "camera_id": "ws-test",
            "fps_target": 2,
        })
        print(f"Job started: {r.status_code} {r.json().get('job_id', 'unknown')}")

        for i in range(20):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(msg)
                t = data.get("type", "-")
                f = data.get("frame_id", "-")
                s = data.get("stage_id", "-")
                lat = data.get("latency_ms")
                thumb = "YES" if data.get("thumbnail_b64") else "no"
                lat_str = f"{lat:.1f}ms" if lat else "-"
                print(f"  Event {i+1:2d}: type={t:20s} stage={s:20s} latency={lat_str:>8s} thumb={thumb}")
            except asyncio.TimeoutError:
                print(f"  Timeout after 10s — no more events")
                break

        # Stop the job
        requests.delete("http://localhost:8000/pipeline/process")
        print("Done — job stopped")

asyncio.run(listen())
