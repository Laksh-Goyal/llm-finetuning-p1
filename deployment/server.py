from fastapi import FastAPI
from vllm import LLM, SamplingParams

app = FastAPI()
llm = LLM(model="models/dpo", tensor_parallel_size=1)

@app.post("/generate")
async def generate(prompt: str):
    params = SamplingParams(max_tokens=256)
    output = llm.generate([prompt], params)
    return {"response": output[0].outputs[0].text}
