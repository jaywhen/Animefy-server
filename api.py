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

def compress_image_b64(b64, mb=190, k=0.9):
    f = base64.b64decode(b64)
    with BytesIO(f) as im:
        o_size = len(im.getvalue()) // 1024
        if o_size <= mb:
            return b64
        im_out = im
        while o_size > mb:
            img = Image.open(im_out)
            x, y = img.size
            out = img.resize((int(x*k), int(y*k)), Image.ANTIALIAS)
            im_out.close()
            im_out = BytesIO()
            out.save(im_out, 'jpeg')
            o_size = len(im_out.getvalue()) // 1024
        b64 = base64.b64encode(im_out.getvalue())
        im_out.close()
        return str(b64, encoding='utf8')

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "This is a FastAPI project!"}

@app.post("/api/anime", response_class=Response)
async def anime(request: Request):
    data = await request.json()
    b64 = compress_image_b64(data['img_base64'])
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    
    with torch.no_grad():
        image = to_tensor(img).unsqueeze(0) * 2 - 1
        out = net(image.to("cpu"), False).cpu()
        out = out.squeeze(0).clip(-1, 1) * 0.5 + 0.5
        out = to_pil_image(out)
    output_buffer = BytesIO()
    out.save(output_buffer, 'jpeg')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str
