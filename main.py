import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from ml_web_inference import expose, Request, StreamingResponse, get_proper_device

from huggingface_hub import hf_hub_download
import torch

from moshi.models import loaders, LMGen

import setproctitle
import argparse
import io
import torchaudio

from download import install_requirements

model_size = 30000
mimi = None
lm_gen = None
device = None


async def inference(request: Request) -> StreamingResponse:
    data = await request.json()
    sample_rate = data["sample_rate"]
    audio_data = data["audio_data"]

    target_sample_rate = 24000
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    wav = torch.tensor(audio_data).unsqueeze(0)
    wav = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(wav)
    # pad to multiple of frame_size
    wav = torch.nn.functional.pad(wav, (0, frame_size - wav.size(-1) % frame_size))
    print("wav shape", wav.shape)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        all_codes = []
        with mimi.streaming(1):
            for offset in range(0, wav.shape[-1], frame_size):
                frame = wav[:, :, offset : offset + frame_size]
                codes = mimi.encode(frame)
                assert codes.shape[-1] == 1, codes.shape
                all_codes.append(codes)
        out_wav_chunks = []
        with lm_gen.streaming(1), mimi.streaming(1):
            for code in all_codes:
                tokens_out = lm_gen.step(code.to(device))
                if tokens_out is not None:
                    wav_chunk = mimi.decode(tokens_out[:, 1:])
                    out_wav_chunks.append(wav_chunk)
        decoded = torch.cat(out_wav_chunks, dim=-1)
        print("decoded shape", decoded.shape)

    result_arr = decoded.squeeze(0).cpu()
    result = io.BytesIO()
    torchaudio.save(result, result_arr, target_sample_rate, format="wav")
    result.seek(0)
    return StreamingResponse(result, media_type="application/octet-stream")


def init():
    global mimi, device, lm_gen
    install_requirements()
    device = get_proper_device(model_size)
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(8)
    moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
    moshi = loaders.get_moshi_lm(moshi_weight, device=device)
    lm_gen = LMGen(moshi, temp=0.8, temp_text=0.7)


def hangup():
    global mimi, lm_gen
    del mimi
    del lm_gen
    torch.cuda.empty_cache()


if __name__ == "__main__":
    setproctitle.setproctitle("moshi-web-inference")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9234)
    parser.add_argument("--api-name", type=str, default="moshi")
    parser.add_argument("--hangup-timeout-sec", type=int, default=900)
    parser.add_argument("--hangup-interval-sec", type=int, default=60)
    args = parser.parse_args()
    expose(
        args.api_name,
        inference,
        port=args.port,
        hangup_timeout_sec=args.hangup_timeout_sec,
        hangup_interval_sec=args.hangup_interval_sec,
        init_function=init,
        hangup_function=hangup,
    )
