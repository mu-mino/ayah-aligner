import os
import sys
import subprocess
import asyncio
from mapping import BASE_DIR
from pathlib import Path

start_id, end_id = 4, 5
cwd = os.getcwd()


async def run_mapping(text_file, audio, video, surah_id):
    process = await asyncio.create_subprocess_exec(
        f"{cwd}/.venv/bin/python3",
        "mapping.py",
        "--text",
        text_file,
        "--audio",
        audio,
        "--video",
        video,
        "--surah",
        str(surah_id),
    )
    await process.wait()
    return process


async def run_ass(mapping, video, output):
    process = await asyncio.create_subprocess_exec(
        f"{cwd}/.venv/bin/python3",
        "modules/text_ass.ju.py",
        mapping,
        video,
        output,
        stdout=sys.stdout,
    )
    await process.wait()
    return process


async def run_burn_subs(video_file, ass_file, audio_dir, out_dir):
    process = await asyncio.create_subprocess_exec(
        f"{cwd}/.venv/bin/python3",
        "burn_subtitles.py",
        "--video-file",
        video_file,
        "--ass-file",
        ass_file,
        "--audio-dir",
        audio_dir,
        "--out-dir",
        out_dir,
    )
    await process.wait()
    return process


async def process_pipeline(i):
    translation_file = subprocess.run(
        f"find /home/muhammed-emin-eser/desk/din/quran/eng_translation/chunked_translation/ -type f -name '{i}_*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cropped_video = subprocess.run(
        f"find /home/muhammed-emin-eser/desk/din/quran/maher_workaround/Quran_cropped/ -type f -name '*({i})*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    audio_dir = "/home/muhammed-emin-eser/desk/din/quran/maher_playlist/maher_playlist/"
    audio = subprocess.run(
        f"find {audio_dir} -type f -name '*({i})*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # 1. Mapping prüfen
    p_mapping = await run_mapping(translation_file, audio, cropped_video, i)
    if p_mapping.returncode != 0:
        raise RuntimeError(
            f"Pipeline {i}: mapping.py fehlgeschlagen mit Exit-Code {p_mapping.returncode}"
        )

    mapping_file = subprocess.run(
        f"find {BASE_DIR}/output/mapping/ -type f -name '{i}_*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    overlay_video = subprocess.run(
        f"find /home/muhammed-emin-eser/desk/din/quran/maher_workaround/with_overlay/ -type f -name '*({i})*'",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # ass_output = Path(translation_file).name
    # ass_path = f"{BASE_DIR}/output/ass/{ass_output.replace('.txt', '.ass')}"

    # 2. ASS-Generierung prüfen
    # p_ass = await run_ass(
    #     mapping=mapping_file,
    #     video=overlay_video,
    #     output=ass_path,
    # )
    # if p_ass.returncode != 0:
    #     raise RuntimeError(
    #         f"Pipeline {i}: text_ass.py fehlgeschlagen mit Exit-Code {p_ass.returncode}"
    #     )

    # 3. Burn Subs prüfen
    # p_burn = await run_burn_subs(
    #     video_file=overlay_video,
    #     ass_file=ass_path,
    #     audio_dir=audio_dir,
    #     out_dir=f"{BASE_DIR}/output/final/",
    # )
    # if p_burn.returncode != 0:
    #     raise RuntimeError(
    #         f"Pipeline {i}: burn_subtitles.py fehlgeschlagen mit Exit-Code {p_burn.returncode}"
    #     )


async def main():
    tasks = [process_pipeline(i) for i in range(start_id, end_id)]
    await asyncio.gather(*tasks)


asyncio.run(main())
