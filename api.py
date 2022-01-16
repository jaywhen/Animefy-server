from PIL import Image
from io import BytesIO
import base64
from model import Generator
import torch
import gc
from torchvision.transforms.functional import to_tensor, to_pil_image
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

gc.collect()

net = Generator()
net.load_state_dict(torch.load("./weights/paprika.pt", map_location="cpu"))
net.to("cpu").eval()

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/api/anime", response_class=Response)
async def anime(request: Request):
    data = await request.json()
    img = Image.open(BytesIO(base64.b64decode(data['img_base64']))).convert("RGB")
    with torch.no_grad():
        image = to_tensor(img).unsqueeze(0) * 2 - 1
        out = net(image.to("cpu"), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)
    output_buffer = BytesIO()
    out.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str
