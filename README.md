A custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that allows you to tag and save generated audio (music, voice recordings, sound effects) in various formats and **automatically embed comprehensive metadata (tags)**.
Includes utility nodes for reading tags, inspecting raw metadata, and sending audio directly to Telegram.

Perfect for AI music workflows (e.g., AudioLDM, Stable Audio, Bark), as you can write essential information like the **generation prompt** directly into the audio file as a comment!

## ✨ Features

* **5 Supported Audio Formats:** MP3, FLAC, M4A (AAC), OGG (Vorbis), and AIFF.
* **Comprehensive Tags:** Set Title, Artist, Album, Genre, Year, Publisher, Copyright, Composer, Album Artist, BPM, Initial Key, ISRC, Encoder, Track Number, License, Comments, and even **multi-line Lyrics** directly within ComfyUI.
* **Utility & Reader Nodes:** Easily read, inspect all embedded tags as a text summary, or push generated tracks directly to Telegram channels/chats.
* **Audio Pass-Through:** The audio signal is output directly from the node, allowing you to seamlessly pass it to other nodes (like Preview Audio).
* **Automatic Folder Structure:** Missing subdirectories in the `output` folder are created automatically.

## 📦 Supported Formats & Standards

| Node Name | Format / Type | Tagging Standard / Action | Best for... |
| :--- | :--- | :--- | :--- |
| **Save MP3 with Tags 🎵** | `.mp3` | ID3v2 | Universal compatibility |
| **Save FLAC with Tags 🎶** | `.flac` | Vorbis Comments | Lossless quality |
| **Save M4A with Tags 🍏** | `.m4a` | MP4 Atoms | Apple ecosystem & smartphones |
| **Save OGG with Tags 🌐** | `.ogg` | Vorbis Comments | Web & video games |
| **Save AIFF with Tags 💿** | `.aiff` | ID3v2 | Uncompressed Apple Lossless |
| **Read Audio Tags 🔍** | Utility | Metadata Reader | Extracting tags & embedded cover art |
| **Inspect All Audio Tags 📋** | Utility | Tag Inspector | Debugging & full raw tag text dump |
| **Send Audio to Telegram ✈️** | Utility | Telegram API | Direct sharing of audio and reports via bot |

## 🛠️ Installation

### Method 1: ComfyUI Manager (Recommended)
*(Once the node is registered in the Manager database)*
1. Open the **ComfyUI Manager**.
2. Click on "Install Custom Nodes".
3. Search for `ComfyUI Audio Tagger` and install it.
4. Restart ComfyUI.

### Method 2: Manual via Git
1. Open your terminal and navigate to the `custom_nodes` folder of your ComfyUI installation:
   ```bash
   cd ComfyUI/custom_nodes

2. Clone this repository:
    ```bash
    git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_NAME>/comfyui-audio-tagger.git

3. Install the required dependencies (the mutagen library):

    ```bash
    cd comfyui-audio-tagger
    pip install -r requirements.txt

Important for Windows Portable Users: Use ComfyUI's embedded Python instead:

    ```bash
    ..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
    Restart ComfyUI.

### ⚠️ Requirements & Notes
Mutagen: This node uses the Python library mutagen to write metadata. It is installed automatically via requirements.txt.

FFmpeg: To save compressed formats like MP3 or M4A, the underlying torchaudio requires an FFmpeg backend installed on your system. This is usually the case for most ComfyUI setups anyway (required for video nodes, for example). If your system lacks FFmpeg, the node safely throws a warning in the console instead of crashing, and passes the audio through unsaved.

## 🚀 Usage in Workflow
Right-click on the canvas (or double-click).

Navigate to audio -> Save MP3 with Tags 🎵 (or any of the other formats).

Connect the audio output of your generation node to the input of the tagger node.

Fill out the text fields as desired. (Tip: You can right-click the node to convert text widgets to inputs, allowing you to pass prompts from an LLM node directly into the comment field!)

Created for the ComfyUI community. Happy generating and tagging!
