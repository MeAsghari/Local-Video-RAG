import struct
import numpy as np
import torch
import webbrowser
import os
from transformers import AutoTokenizer, CLIPTextModelWithProjection

# --- UI Libraries ---
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import print as rprint

# --- Configuration ---
DB_FILE = "video_index.bin"
VIDEO_FILE = "test.mp4" # Video filename located next to the program
MODEL_ID = "openai/clip-vit-base-patch32"
THRESHOLD = 0.23

# Initialize rich console
console = Console()

def create_html_player(video_path, timestamp, query):
    """
    Create a simple HTML file that plays the video at the desired second
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Search Result</title>
        <style>
            body {{ font-family: sans-serif; background-color: #1a1a1a; color: white; text-align: center; padding: 20px; }}
            video {{ width: 80%; max-width: 800px; border: 2px solid #00ff99; border-radius: 10px; }}
            .info {{ margin-top: 20px; font-size: 1.2em; }}
            .highlight {{ color: #00ff99; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>🔍 Search Result: <span class="highlight">{query}</span></h1>
        
        <video id="player" controls autoplay>
            <source src="{video_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>

        <div class="info">
            Jumped to: <span class="highlight">{int(timestamp // 60):02d}:{int(timestamp % 60):02d}</span>
        </div>

        <script>
            var video = document.getElementById("player");
            // Ensure video starts from the correct time
            video.currentTime = {timestamp};
        </script>
    </body>
    </html>
    """
    
    with open("result_player.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Automatically open the file in the browser
    webbrowser.open("file://" + os.path.realpath("result_player.html"))


# --- Loading Models with Elegant Effect ---
with console.status("[bold green]Loading AI Models (CLIP)...", spinner="dots"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_ID)

def load_index(filename):
    data = []
    vector_size = 512
    entry_size = 8 + (vector_size * 4)
    try:
        with open(filename, "rb") as f:
            while True:
                chunk = f.read(entry_size)
                if not chunk: break
                timestamp = struct.unpack("d", chunk[:8])[0]
                vector = np.frombuffer(chunk[8:], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0: vector = vector / norm
                data.append({"time": timestamp, "vector": vector})
            return data
    except FileNotFoundError:
        return []

def search_smart_window(query, index_data):
    # 1. Convert text to vector
    inputs = tokenizer([query], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
    text_embed = outputs.text_embeds.numpy()[0]
    text_embed /= np.linalg.norm(text_embed)
    
    # 2. Calculate scores
    raw_scores = []
    timestamps = []
    for entry in index_data:
        score = np.dot(text_embed, entry["vector"])
        raw_scores.append(score)
        timestamps.append(entry["time"])
    
    raw_scores = np.array(raw_scores)
    
    # 3. Smoothing (Moving Average)
    window_size = 3
    smoothed_scores = np.convolve(raw_scores, np.ones(window_size)/window_size, mode='same')

    # 4. Calculate Softmax
    logit_scale = 100.0
    exp_scores = np.exp(smoothed_scores * logit_scale)
    probs = exp_scores / np.sum(exp_scores)
    
    best_idx = np.argmax(probs)
    
    # Calculate regional confidence
    region_confidence = probs[best_idx]
    if best_idx > 0: region_confidence += probs[best_idx-1]
    if best_idx < len(probs)-1: region_confidence += probs[best_idx+1]
    
    return timestamps[best_idx], region_confidence, raw_scores[best_idx]

# --- Start Program ---
console.print(Panel.fit("[bold cyan]🎬 Edge Video Search Engine[/bold cyan]\n[dim]Powered by CLIP & C++ Indexing[/dim]"))

index = load_index(DB_FILE)
console.print(f"[bold green]✅ Index Loaded:[/bold green] {len(index)} frames indexed.\n")

while True:
    query = Prompt.ask("[bold yellow]🔍 What are you looking for?[/bold yellow] (or 'q' to quit)")
    if query.lower() == 'q': break
    
    full_query = f"a photo of a {query}"
    
    # Search
    best_time, confidence, raw_score = search_smart_window(full_query, index)
    
    minutes = int(best_time // 60)
    seconds = int(best_time % 60)
    time_str = f"{minutes:02d}:{seconds:02d}"

    # Display results in table
    table = Table(title=f"Results for '{query}'")
    table.add_column("Attribute", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Timestamp", time_str)
    table.add_row("Raw Similarity", f"{raw_score:.4f}")
    
    # Change color based on confidence
    conf_color = "green" if raw_score >= THRESHOLD else "red"
    status_icon = "✅" if raw_score >= THRESHOLD else "⚠️"
    
    table.add_row("Confidence", f"[{conf_color}]{confidence*100:.2f}%[/]")
    table.add_row("Status", f"{status_icon} {'Match Found' if raw_score >= THRESHOLD else 'Low Confidence'}")
    
    console.print(table)

    if raw_score >= THRESHOLD:
        # Here comes the cool HTML player part
        rprint(f"\n[bold green]🚀 Opening player at {time_str}...[/bold green]")
        create_html_player(VIDEO_FILE, best_time, query)
    else:
        rprint("\n[bold red]❌ Score too low. Not opening player.[/bold red]")
    
    print("-" * 40)