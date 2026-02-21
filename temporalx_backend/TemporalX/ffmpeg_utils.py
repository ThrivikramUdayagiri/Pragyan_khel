import subprocess

def fix_mp4_fps_metadata(input_path: str, output_path: str, fps: int) -> bool:
    """
    Use ffmpeg to rewrite the MP4 container metadata to set the correct FPS.
    Returns True if successful, False otherwise.
    """
    try:
        # -y: overwrite output, -loglevel error: only show errors
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c', 'copy', '-map', '0',
            '-r', str(fps),
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0
    except Exception:
        return False
