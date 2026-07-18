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
    elif batch.status == "failed":
        print("fail")


async def main():
    client = OpenAI(
        api_key="sk-ws-H.XIMXYL.H7cm.MEUCIHJBgnnF2vQkc4mhOXKbWHYfhHvA85gsPl0BGcydlUBhAiEA4uFwb25pzNUQVTqNSOOtDjRnDRUQQJ75uJ4o_8rGywA",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    for file in glob.glob(r"/home/muhammed-emin-eser/desk/din/quran/qwen_res_jsonl/*"):
        if file.replace(
            "/home/muhammed-emin-eser/desk/din/quran/qwen_res_jsonl/", ""
        ).replace(".api", "") not in [
            x.replace(
                "/home/muhammed-emin-eser/desk/din/quran/qwen_final_jsonl/", ""
            ).replace(".jsonl", "")
            for x in glob.glob(
                r"/home/muhammed-emin-eser/desk/din/quran/qwen_final_jsonl/*"
            )
        ]:
            print(file)
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
