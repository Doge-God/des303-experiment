from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich.align import Align
import keyboard
from voice_util_test import STT
from rich import box
from rich import print

# STT setup
stt = STT(vad_threshold=0.7, vad_silence=2) 
print("Spawned STT")


# Setup progress bars
speech_progress = Progress(
    TextColumn("[bold blue]Speech Prob.:"),
    BarColumn(bar_width=40, complete_style="bold blue", finished_style="blue", pulse_style="blue"),
    TextColumn("{task.percentage:>3.0f}%"),
    TextColumn("{task.fields[raw]:.2f}"),
    expand=True
)

vad_progress = Progress(
    TextColumn("[bold green]VAD val:"),  # Pad label to align with Speech Prob.
    BarColumn(bar_width=40, complete_style="bold green", finished_style="green", pulse_style="green"),
    TextColumn("{task.percentage:>3.0f}%"),
    TextColumn("{task.fields[raw]:.2f}"),
    expand=True
)

speech_task = speech_progress.add_task("speech", total=1.0, raw=0.0)
vad_task = vad_progress.add_task("vad", total=1.0, raw=0.0)
result_text = Text("Result: ", style="bold white")

# Update callbacks
def on_speech_prob_change(prob):
    speech_progress.update(speech_task, completed=prob, raw=prob)
    live.update(render_status())  

def on_vad_val_change(val):
    vad_progress.update(vad_task, completed=val, raw=val)
    live.update(render_status())

def on_transcription_change(text):
    result_text.plain = f"Result: {text.strip()}"
    live.update(render_status())

# Status rendering
def render_status():
    table = Table.grid(padding=(0, 1), expand=False)
    table.width = 75 
    table.box = box.SQUARE
    table.add_row(Align.center(speech_progress, vertical="middle"))
    table.add_row(Align.center(vad_progress, vertical="middle"))  
    table.add_row(Panel(result_text, title="Transctiption", border_style="bold white"))
    return table

while True:
    # Display with Live
    print("Press SPACE to start transcription...")
    keyboard.wait('space')
    print("Starting transcription...")
    with Live(render_status(), refresh_per_second=60) as live: 
        try:
            on_transcription_change("")
            result = stt.transcribe_stream(
                on_vad_val_change=on_vad_val_change,
                on_streaming_transctiption_change=on_transcription_change, 
                on_vad_prob_change=on_speech_prob_change
            )  
            on_transcription_change(result)
        except KeyboardInterrupt:
            print("Stopping...")
            stt.stop()
