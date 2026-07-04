from pathlib import Path
from openai import OpenAI
import asyncio
import glob


async def get_res(client, batch_id, id):
    batch = client.batches.retrieve(f"{batch_id}")
    if batch.status == "completed":
        out = batch.output_file_id
        content = client.files.content(f"{out}")
        content.write_to_file(
            f"/home/muhammed-emin-eser/desk/din/quran/qwen_final_jsonl/{id}.jsonl"
        )
        print(content)


async def main():
    client = OpenAI(
        api_key="sk-ws-H.YHLXED.udgC.MEUCIQCZkSkP1JAqaQQFwY75UDkQhIlfwQ_UwnDWaWCxXW3bXwIgMXqQ-vcLhiU5MgmZABvOtz579qTly1Z-RS2RTmW06xc",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    for file in glob.glob(r"/home/muhammed-emin-eser/desk/din/quran/qwen_res_jsonl/*"):
        with open(file, "r") as f:
            batch_id = f.readlines()
        await get_res(
            client,
            " ".join(batch_id),
            file.replace(
                "/home/muhammed-emin-eser/desk/din/quran/qwen_res_jsonl/", ""
            ).replace(".api", ""),
        )


if __name__ == "__main__":
    asyncio.run(main())
