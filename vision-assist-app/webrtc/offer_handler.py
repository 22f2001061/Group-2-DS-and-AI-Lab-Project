import asyncio
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import Body
from pydantic import BaseModel
from inference.track_processor import InferenceVideoTrack

pcs = set()

class Offer(BaseModel):
    sdp: str
    type: str


async def offer(sdp: Offer = Body(...)):
    pc = RTCPeerConnection()
    pcs.add(pc)
    data_channel_holder = {"channel": None}

    @pc.on("datachannel")
    def on_datachannel(channel):
        data_channel_holder["channel"] = channel

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(InferenceVideoTrack(track, data_channel_holder))

    offer_desc = RTCSessionDescription(sdp=sdp.sdp, type=sdp.type)
    await pc.setRemoteDescription(offer_desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    async def cleanup():
        await asyncio.sleep(1800)
        if pc in pcs:
            await pc.close()
            pcs.discard(pc)

    asyncio.ensure_future(cleanup())

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
