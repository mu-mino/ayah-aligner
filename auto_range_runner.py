import os
import subprocess
import asyncio

start_id, end_id = 98, 99

cwd = os.getcwd()

for i in (
    range(start_id, end_id) if start_id is not end_id else (start_id, start_id + 1)
):
    text_file = subprocess.run(
        f"find /home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation/ -type f -name '{i}_*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    video = subprocess.run(
        f"find /home/muhammed-emin-eser/desk/din/quran/maher_workaround/Quran_cropped/ -type f -name '*({i})*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    audio = subprocess.run(
        f"find /home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist/ -type f -name '*({i})*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    async def run_process(text_file, audio, video):
        process = await asyncio.create_subprocess_exec(
            f"{cwd}/.venv/bin/python3",
            "main.py",
            "--text",
            text_file,
            "--audio",
            audio,
            "--video",
            video,
            stdout=None,
            stderr=None,
        )
        stdout, stderr = await process.communicate()

    asyncio.run(run_process(text_file, audio, video))
    print(f"running process with id {i}")
