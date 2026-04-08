import os
import subprocess
import asyncio

start_id, end_id = 74, 75
cwd = os.getcwd()

async def run_process(text_file, audio, video, surah_id):
    process = await asyncio.create_subprocess_exec(
        f"{cwd}/.venv/bin/python3",
        "main.py",
        "--text",
        text_file,
        "--audio",
        audio,
        "--video",
        video,
        "--surah",
        str(surah_id),
    )
    return process


async def main():
    processes = []

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

        process = await run_process(text_file, audio, video, i)
        processes.append(process)
        print(f"{'#'*20} RUNNING COMMAND: {'#'*20}")
        print(
            f"python main.py --video '{video}' --text '{text_file}' --surah '{i}' --audio '{audio}'"
        )
        print("#" * 58)
        print(f"running process with id {i}")
        print("#" * 58)

        is_last = i == end_id - 1
        if len(processes) == 13 or (is_last and processes):
            tasks = {asyncio.create_task(p.wait()): p for p in processes}
            done, pending = await asyncio.wait(
                tasks.keys(), return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                proc = tasks.pop(task)
                processes.remove(proc)
                print(f"Prozess fertig: PID={proc.pid}")


asyncio.run(main())
