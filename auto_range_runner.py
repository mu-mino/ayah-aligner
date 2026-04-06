import os
import subprocess
import asyncio
import time

start_id, end_id = 98, 99

cwd = os.getcwd()


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


def wait_until_stopped(pid):
    while is_running(pid):
        time.sleep(5)


async def check_pids(pids):
    tasks = [asyncio.create_task(wait_until_stopped(pid)) for pid in pids]
    await asyncio.gather(*tasks)


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
    return process


pids = []

for proc, i in enumerate(range(start_id, end_id)):
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

    process = asyncio.run(run_process(text_file, audio, video))
    pids.append(process.pid)

    print(f"running process with id {i}")
    if len(pids) == 13:
        asyncio.run(check_pids(pids))
        pids = []
    elif len(pids) == proc + 1:
        asyncio.run(check_pids(pids))
