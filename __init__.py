import os
import torch
import torchaudio
import folder_paths
import io
import base64
import random
import numpy as np
from PIL import Image
import requests

# Mutagen generic File for reading
from mutagen import File

# MP3 & AIFF imports (ID3 Standard)
from mutagen.id3 import (
    ID3, TALB, TIT2, TPE1, TCON, TDRC, COMM, USLT, TXXX, APIC, 
    TPUB, TCOP, TBPM, TCOM, TPE2, TKEY, TSRC, TSSE, TRCK
)
from mutagen.mp3 import MP3
from mutagen.aiff import AIFF

# FLAC & OGG imports (Vorbis Comments & Picture)
from mutagen.flac import FLAC, Picture
from mutagen.oggvorbis import OggVorbis

# M4A imports (MP4 Atoms)
from mutagen.mp4 import MP4, MP4Cover

# ==========================================
# LICENSE OPTIONS FOR DROPDOWN
# ==========================================
LICENSE_OPTIONS = [
    "CC-BY 4.0 (Attribution - Credit required)",
    "CC0 1.0 (Public Domain - Free for any use)",
    "CC-BY-SA 4.0 (ShareAlike - Credit + adapt under same license)",
    "CC-BY-NC 4.0 (NonCommercial - Credit + no commercial use)",
    "CC-BY-NC-SA 4.0 (NonCom + ShareAlike - No commercial + adapt under same)",
    "CC-BY-ND 4.0 (NoDerivs - Credit + no modifications)",
    "All Rights Reserved (Copyrighted)",
    "Custom (See Comments)"
]

# Comprehensive Telegram sending modes and their corresponding API methods
SEND_MODES = ["audio", "voice", "document", "photo", "video", "animation", "video_note", "message"]
TELEGRAM_METHODS = {
    "audio": "sendAudio",
    "voice": "sendVoice",
    "document": "sendDocument",
    "photo": "sendPhoto",
    "video": "sendVideo",
    "animation": "sendAnimation",
    "video_note": "sendVideoNote",
    "message": "sendMessage"
}

# ==========================================
# HELPERS: IMAGE CONVERSIONS
# ==========================================
def tensor_to_jpeg_bytes(image_tensor):
    """Converts a ComfyUI image tensor to JPEG bytes for embedding as cover art."""
    if image_tensor is None:
        return None
    i = 255. * image_tensor[0].cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=90)
    return img_byte_arr.getvalue()

def jpeg_bytes_to_tensor(img_bytes):
    """Converts raw image bytes from audio tags into a ComfyUI image tensor."""
    if img_bytes is None:
        return None
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None, :, :, :] # Shape [1, H, W, C]
        return img_tensor
    except Exception as e:
        print(f"[Tag Reader Error] Failed to parse cover image: {e}")
        return None

# ==========================================
# 1. MP3 NODE
# ==========================================
class SaveMP3WithTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI_MP3"}),
                "title": ("STRING", {"default": "My Title"}),
                "artist": ("STRING", {"default": "My Artist"}),
                "album": ("STRING", {"default": "My Album"}),
                "genre": ("STRING", {"default": "My Genre"}),
                "year": ("STRING", {"default": "2026"}),
                "publisher": ("STRING", {"default": "My Publisher"}),
                "copyright": ("STRING", {"default": "My Copyright"}),
                "composer": ("STRING", {"default": "My Composer"}),
                "album_artist": ("STRING", {"default": "My Album Artist"}),
                "bpm": ("STRING", {"default": "120"}),
                "initial_key": ("STRING", {"default": "C maj"}),
                "isrc": ("STRING", {"default": "My ISRC"}),
                "encoder": ("STRING", {"default": "ComfyUI Audio Tagger"}),
                "track_number": ("STRING", {"default": "1"}),
                "license": (LICENSE_OPTIONS, {"default": LICENSE_OPTIONS[0]}),
                "comment": ("STRING", {"default": "Generated with ComfyUI", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "cover_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "save_mp3"
    OUTPUT_NODE = True
    CATEGORY = "Audio Tagger"

    def save_mp3(self, audio, filename_prefix, title, artist, album, genre, year, publisher, copyright, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, license, comment, lyrics, cover_image=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, audio["waveform"].shape[2], audio["sample_rate"]
        )
        
        os.makedirs(full_output_folder, exist_ok=True)
        file_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}.mp3")
        
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        try:
            torchaudio.save(file_path, waveform, sample_rate, format="mp3")
        except Exception as e:
            print(f"[MP3 Error] Could not save MP3: {e}")
            return {"ui": {}, "result": (audio, "")}
        
        try:
            audio_file = MP3(file_path, ID3=ID3)
            if audio_file.tags is None:
                audio_file.add_tags()
            
            audio_file.tags.add(TIT2(encoding=3, text=title))
            audio_file.tags.add(TPE1(encoding=3, text=artist))
            audio_file.tags.add(TALB(encoding=3, text=album))
            audio_file.tags.add(TCON(encoding=3, text=genre))
            audio_file.tags.add(TDRC(encoding=3, text=year))
            
            if publisher and publisher.strip():
                audio_file.tags.add(TPUB(encoding=3, text=publisher))
            if copyright and copyright.strip():
                audio_file.tags.add(TCOP(encoding=3, text=copyright))
            if composer and composer.strip():
                audio_file.tags.add(TCOM(encoding=3, text=composer))
            if album_artist and album_artist.strip():
                audio_file.tags.add(TPE2(encoding=3, text=album_artist))
            if bpm and bpm.strip():
                audio_file.tags.add(TBPM(encoding=3, text=bpm))
            if initial_key and initial_key.strip():
                audio_file.tags.add(TKEY(encoding=3, text=initial_key))
            if isrc and isrc.strip():
                audio_file.tags.add(TSRC(encoding=3, text=isrc))
            if encoder and encoder.strip():
                audio_file.tags.add(TSSE(encoding=3, text=encoder))
            if track_number and track_number.strip():
                audio_file.tags.add(TRCK(encoding=3, text=track_number))
            if license and license.strip():
                audio_file.tags.add(TXXX(encoding=3, desc='License', text=license))
            if comment and comment.strip():
                audio_file.tags.add(COMM(encoding=3, lang='eng', desc='Comment', text=comment))
            if lyrics and lyrics.strip():
                audio_file.tags.add(USLT(encoding=3, lang='eng', desc='Lyrics', text=lyrics))
                
            if cover_image is not None:
                img_bytes = tensor_to_jpeg_bytes(cover_image)
                audio_file.tags.add(APIC(encoding=3, mime='image/jpeg', type=3, desc='Cover', data=img_bytes))
            
            audio_file.save(v2_version=3)
            print(f"[Audio Tagger] Successfully saved: {file_path}")
        except Exception as e:
            print(f"[Audio Tagger Error] Tagging failed: {e}")
            
        return {"ui": {"audio": [{"filename": os.path.basename(file_path), "subfolder": subfolder, "type": "output"}]}, "result": (audio, file_path)}


# ==========================================
# 2. FLAC NODE
# ==========================================
class SaveFLACWithTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI_FLAC"}),
                "title": ("STRING", {"default": "My Title"}),
                "artist": ("STRING", {"default": "My Artist"}),
                "album": ("STRING", {"default": "My Album"}),
                "genre": ("STRING", {"default": "My Genre"}),
                "year": ("STRING", {"default": "2026"}),
                "publisher": ("STRING", {"default": "My Publisher"}),
                "copyright": ("STRING", {"default": "My Copyright"}),
                "composer": ("STRING", {"default": "My Composer"}),
                "album_artist": ("STRING", {"default": "My Album Artist"}),
                "bpm": ("STRING", {"default": "120"}),
                "initial_key": ("STRING", {"default": "C maj"}),
                "isrc": ("STRING", {"default": "My ISRC"}),
                "encoder": ("STRING", {"default": "ComfyUI Audio Tagger"}),
                "track_number": ("STRING", {"default": "1"}),
                "license": (LICENSE_OPTIONS, {"default": LICENSE_OPTIONS[0]}),
                "comment": ("STRING", {"default": "Generated with ComfyUI", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "cover_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "save_flac"
    OUTPUT_NODE = True
    CATEGORY = "Audio Tagger"

    def save_flac(self, audio, filename_prefix, title, artist, album, genre, year, publisher, copyright, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, license, comment, lyrics, cover_image=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, audio["waveform"].shape[2], audio["sample_rate"]
        )
        
        os.makedirs(full_output_folder, exist_ok=True)
        file_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}.flac")
        
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        torchaudio.save(file_path, waveform, sample_rate, format="flac")
        
        try:
            dirty_id3 = ID3(file_path)
            dirty_id3.delete()
        except Exception:
            pass
            
        try:
            audio_file = FLAC(file_path)
            if audio_file.tags is None:
                audio_file.add_tags()
            
            audio_file["TITLE"] = [title]
            audio_file["ARTIST"] = [artist]
            audio_file["ALBUM"] = [album]
            audio_file["GENRE"] = [genre]
            audio_file["DATE"] = [year]
            
            if publisher and publisher.strip():
                audio_file["PUBLISHER"] = [publisher]
            if copyright and copyright.strip():
                audio_file["COPYRIGHT"] = [copyright]
            if composer and composer.strip():
                audio_file["COMPOSER"] = [composer]
            if album_artist and album_artist.strip():
                audio_file["ALBUMARTIST"] = [album_artist]
            if bpm and bpm.strip():
                audio_file["BPM"] = [bpm]
            if initial_key and initial_key.strip():
                audio_file["INITIALKEY"] = [initial_key]
            if isrc and isrc.strip():
                audio_file["ISRC"] = [isrc]
            if encoder and encoder.strip():
                audio_file["ENCODER"] = [encoder]
            if track_number and track_number.strip():
                audio_file["TRACKNUMBER"] = [track_number]
            if license and license.strip(): 
                audio_file["LICENSE"] = [license]
            if comment and comment.strip(): 
                audio_file["COMMENT"] = [comment]
                audio_file["DESCRIPTION"] = [comment]
            if lyrics and lyrics.strip(): 
                audio_file["LYRICS"] = [lyrics]
                audio_file["UNSYNCEDLYRICS"] = [lyrics]
                
            if cover_image is not None:
                img_bytes = tensor_to_jpeg_bytes(cover_image)
                pic = Picture()
                pic.type = 3
                pic.mime = "image/jpeg"
                pic.desc = "Cover"
                pic.data = img_bytes
                audio_file.add_picture(pic)
                
            audio_file.save()
            print(f"[Audio Tagger] Successfully saved: {file_path}")
        except Exception as e:
            print(f"[Audio Tagger Error] Tagging failed: {e}")
            
        return {"ui": {"audio": [{"filename": os.path.basename(file_path), "subfolder": subfolder, "type": "output"}]}, "result": (audio, file_path)}


# ==========================================
# 3. M4A / AAC NODE
# ==========================================
class SaveM4AWithTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI_M4A"}),
                "title": ("STRING", {"default": "My Title"}),
                "artist": ("STRING", {"default": "My Artist"}),
                "album": ("STRING", {"default": "My Album"}),
                "genre": ("STRING", {"default": "My Genre"}),
                "year": ("STRING", {"default": "2026"}),
                "publisher": ("STRING", {"default": "My Publisher"}),
                "copyright": ("STRING", {"default": "My Copyright"}),
                "composer": ("STRING", {"default": "My Composer"}),
                "album_artist": ("STRING", {"default": "My Album Artist"}),
                "bpm": ("STRING", {"default": "120"}),
                "initial_key": ("STRING", {"default": "C maj"}),
                "isrc": ("STRING", {"default": "My ISRC"}),
                "encoder": ("STRING", {"default": "ComfyUI Audio Tagger"}),
                "track_number": ("STRING", {"default": "1"}),
                "license": (LICENSE_OPTIONS, {"default": LICENSE_OPTIONS[0]}),
                "comment": ("STRING", {"default": "Generated with ComfyUI", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "cover_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "save_m4a"
    OUTPUT_NODE = True
    CATEGORY = "Audio Tagger"

    def save_m4a(self, audio, filename_prefix, title, artist, album, genre, year, publisher, copyright, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, license, comment, lyrics, cover_image=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, audio["waveform"].shape[2], audio["sample_rate"]
        )
        
        os.makedirs(full_output_folder, exist_ok=True)
        file_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}.m4a")
        
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        try:
            torchaudio.save(file_path, waveform, sample_rate, format="mp4")
        except Exception as e:
            print(f"[M4A Error] Could not save M4A: {e}")
            return {"ui": {}, "result": (audio, "")}
        
        try:
            audio_file = MP4(file_path)
            if audio_file.tags is None:
                audio_file.add_tags()
                
            audio_file['©nam'] = [title]
            audio_file['©ART'] = [artist]
            audio_file['©alb'] = [album]
            audio_file['©gen'] = [genre]
            audio_file['©day'] = [year]
            
            if publisher and publisher.strip():
                audio_file['©pub'] = [publisher]
            if copyright and copyright.strip():
                audio_file['©cpy'] = [copyright]
            if composer and composer.strip():
                audio_file['©wrt'] = [composer]
            if album_artist and album_artist.strip():
                audio_file['aART'] = [album_artist]
            if bpm and bpm.strip():
                try:
                    audio_file['tmpo'] = [int(bpm)]
                except ValueError:
                    pass
            if initial_key and initial_key.strip():
                audio_file['----:com.apple.iTunes:INITIALKEY'] = [initial_key.encode('utf-8')]
            if isrc and isrc.strip():
                audio_file['©isr'] = [isrc]
            if encoder and encoder.strip():
                audio_file['©too'] = [encoder]
            if track_number and track_number.strip():
                try:
                    t_num = int(track_number.split('/')[0])
                    audio_file['trkn'] = [(t_num, 0)]
                except ValueError:
                    pass
            
            full_comment = comment
            if license and license.strip(): full_comment = f"License: {license} | {comment}"
            if full_comment and full_comment.strip(): 
                audio_file['©cmt'] = [full_comment]
                audio_file['desc'] = [full_comment]
            if lyrics and lyrics.strip(): 
                audio_file['©lyr'] = [lyrics]
                
            if cover_image is not None:
                img_bytes = tensor_to_jpeg_bytes(cover_image)
                audio_file["covr"] = [MP4Cover(img_bytes, imageformat=MP4Cover.FORMAT_JPEG)]
                
            audio_file.save()
            print(f"[Audio Tagger] Successfully saved: {file_path}")
        except Exception as e:
            print(f"[Audio Tagger Error] Tagging failed: {e}")
            
        return {"ui": {"audio": [{"filename": os.path.basename(file_path), "subfolder": subfolder, "type": "output"}]}, "result": (audio, file_path)}


# ==========================================
# 4. OGG / VORBIS NODE
# ==========================================
class SaveOGGWithTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI_OGG"}),
                "title": ("STRING", {"default": "My Title"}),
                "artist": ("STRING", {"default": "My Artist"}),
                "album": ("STRING", {"default": "My Album"}),
                "genre": ("STRING", {"default": "My Genre"}),
                "year": ("STRING", {"default": "2026"}),
                "publisher": ("STRING", {"default": "My Publisher"}),
                "copyright": ("STRING", {"default": "My Copyright"}),
                "composer": ("STRING", {"default": "My Composer"}),
                "album_artist": ("STRING", {"default": "My Album Artist"}),
                "bpm": ("STRING", {"default": "120"}),
                "initial_key": ("STRING", {"default": "C maj"}),
                "isrc": ("STRING", {"default": "My ISRC"}),
                "encoder": ("STRING", {"default": "ComfyUI Audio Tagger"}),
                "track_number": ("STRING", {"default": "1"}),
                "license": (LICENSE_OPTIONS, {"default": LICENSE_OPTIONS[0]}),
                "comment": ("STRING", {"default": "Generated with ComfyUI", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "cover_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "save_ogg"
    OUTPUT_NODE = True
    CATEGORY = "Audio Tagger"

    def save_ogg(self, audio, filename_prefix, title, artist, album, genre, year, publisher, copyright, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, license, comment, lyrics, cover_image=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, audio["waveform"].shape[2], audio["sample_rate"]
        )
        
        os.makedirs(full_output_folder, exist_ok=True)
        file_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}.ogg")
        
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        torchaudio.save(file_path, waveform, sample_rate, format="ogg")
        
        try:
            audio_file = OggVorbis(file_path)
            if audio_file.tags is None:
                audio_file.add_tags()
                
            audio_file["TITLE"] = [title]
            audio_file["ARTIST"] = [artist]
            audio_file["ALBUM"] = [album]
            audio_file["GENRE"] = [genre]
            audio_file["DATE"] = [year]
            
            if publisher and publisher.strip():
                audio_file["PUBLISHER"] = [publisher]
            if copyright and copyright.strip():
                audio_file["COPYRIGHT"] = [copyright]
            if composer and composer.strip():
                audio_file["COMPOSER"] = [composer]
            if album_artist and album_artist.strip():
                audio_file["ALBUMARTIST"] = [album_artist]
            if bpm and bpm.strip():
                audio_file["BPM"] = [bpm]
            if initial_key and initial_key.strip():
                audio_file["INITIALKEY"] = [initial_key]
            if isrc and isrc.strip():
                audio_file["ISRC"] = [isrc]
            if encoder and encoder.strip():
                audio_file["ENCODER"] = [encoder]
            if track_number and track_number.strip():
                audio_file["TRACKNUMBER"] = [track_number]
            if license and license.strip(): 
                audio_file["LICENSE"] = [license]
            if comment and comment.strip(): 
                audio_file["COMMENT"] = [comment]
                audio_file["DESCRIPTION"] = [comment]
            if lyrics and lyrics.strip(): 
                audio_file["LYRICS"] = [lyrics]
                audio_file["UNSYNCEDLYRICS"] = [lyrics]
                
            if cover_image is not None:
                img_bytes = tensor_to_jpeg_bytes(cover_image)
                pic = Picture()
                pic.type = 3
                pic.mime = "image/jpeg"
                pic.desc = "Cover"
                pic.data = img_bytes
                
                pic_data = pic.write()
                encoded_data = base64.b64encode(pic_data).decode("ascii")
                audio_file["metadata_block_picture"] = [encoded_data]
                
            audio_file.save()
            print(f"[Audio Tagger] Successfully saved: {file_path}")
        except Exception as e:
            print(f"[Audio Tagger Error] Tagging failed: {e}")
            
        return {"ui": {"audio": [{"filename": os.path.basename(file_path), "subfolder": subfolder, "type": "output"}]}, "result": (audio, file_path)}


# ==========================================
# 5. AIFF NODE
# ==========================================
class SaveAIFFWithTags:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI_AIFF"}),
                "title": ("STRING", {"default": "My Title"}),
                "artist": ("STRING", {"default": "My Artist"}),
                "album": ("STRING", {"default": "My Album"}),
                "genre": ("STRING", {"default": "My Genre"}),
                "year": ("STRING", {"default": "2026"}),
                "publisher": ("STRING", {"default": "My Publisher"}),
                "copyright": ("STRING", {"default": "My Copyright"}),
                "composer": ("STRING", {"default": "My Composer"}),
                "album_artist": ("STRING", {"default": "My Album Artist"}),
                "bpm": ("STRING", {"default": "120"}),
                "initial_key": ("STRING", {"default": "C maj"}),
                "isrc": ("STRING", {"default": "My ISRC"}),
                "encoder": ("STRING", {"default": "ComfyUI Audio Tagger"}),
                "track_number": ("STRING", {"default": "1"}),
                "license": (LICENSE_OPTIONS, {"default": LICENSE_OPTIONS[0]}),
                "comment": ("STRING", {"default": "Generated with ComfyUI", "multiline": True}),
                "lyrics": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "cover_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "file_path")
    FUNCTION = "save_aiff"
    OUTPUT_NODE = True
    CATEGORY = "Audio Tagger"

    def save_aiff(self, audio, filename_prefix, title, artist, album, genre, year, publisher, copyright, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, license, comment, lyrics, cover_image=None):
        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, audio["waveform"].shape[2], audio["sample_rate"]
        )
        
        os.makedirs(full_output_folder, exist_ok=True)
        file_path = os.path.join(full_output_folder, f"{filename}_{counter:05d}.aiff")
        
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]
        
        try:
            torchaudio.save(file_path, waveform, sample_rate, format="aiff")
        except Exception as e:
            print(f"[AIFF Error] Could not save AIFF: {e}")
            return {"ui": {}, "result": (audio, "")}
        
        try:
            audio_file = AIFF(file_path)
            if audio_file.tags is None:
                audio_file.add_tags()
            
            audio_file.tags.add(TIT2(encoding=3, text=title))
            audio_file.tags.add(TPE1(encoding=3, text=artist))
            audio_file.tags.add(TALB(encoding=3, text=album))
            audio_file.tags.add(TCON(encoding=3, text=genre))
            audio_file.tags.add(TDRC(encoding=3, text=year))
            
            if publisher and publisher.strip():
                audio_file.tags.add(TPUB(encoding=3, text=publisher))
            if copyright and copyright.strip():
                audio_file.tags.add(TCOP(encoding=3, text=copyright))
            if composer and composer.strip():
                audio_file.tags.add(TCOM(encoding=3, text=composer))
            if album_artist and album_artist.strip():
                audio_file.tags.add(TPE2(encoding=3, text=album_artist))
            if bpm and bpm.strip():
                audio_file.tags.add(TBPM(encoding=3, text=bpm))
            if initial_key and initial_key.strip():
                audio_file.tags.add(TKEY(encoding=3, text=initial_key))
            if isrc and isrc.strip():
                audio_file.tags.add(TSRC(encoding=3, text=isrc))
            if encoder and encoder.strip():
                audio_file.tags.add(TSSE(encoding=3, text=encoder))
            if track_number and track_number.strip():
                audio_file.tags.add(TRCK(encoding=3, text=track_number))
            if license and license.strip():
                audio_file.tags.add(TXXX(encoding=3, desc='License', text=license))
            if comment and comment.strip():
                audio_file.tags.add(COMM(encoding=3, lang='eng', desc='Comment', text=comment))
            if lyrics and lyrics.strip():
                audio_file.tags.add(USLT(encoding=3, lang='eng', desc='Lyrics', text=lyrics))
                
            if cover_image is not None:
                img_bytes = tensor_to_jpeg_bytes(cover_image)
                audio_file.tags.add(APIC(encoding=3, mime='image/jpeg', type=3, desc='Cover', data=img_bytes))
            
            audio_file.save(v2_version=3)
            print(f"[Audio Tagger] Successfully saved: {file_path}")
        except Exception as e:
            print(f"[Audio Tagger Error] Tagging failed: {e}")
            
        return {"ui": {"audio": [{"filename": os.path.basename(file_path), "subfolder": subfolder, "type": "output"}]}, "result": (audio, file_path)}


# ==========================================
# 6. TELEGRAM SENDER NODE
# ==========================================
class SendAudioToTelegram:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bot_token": ("STRING", {"default": "YOUR_BOT_TOKEN"}),
                "chat_id": ("STRING", {"default": "YOUR_CHAT_ID"}),
                "send_mode": (SEND_MODES, {"default": "audio"}),
            },
            "optional": {
                "file_path": ("STRING", {"default": "", "forceInput": True}),
                "topic_id": ("STRING", {"default": ""}), 
                "caption": ("STRING", {"default": "🎵 New track generated!", "multiline": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_telegram"
    OUTPUT_NODE = True
    CATEGORY = "Audio Tagger"

    def send_telegram(self, bot_token, chat_id, send_mode="audio", file_path="", topic_id="", caption=""):
        method = TELEGRAM_METHODS.get(send_mode, "sendAudio")
        url = f"https://api.telegram.org/bot{bot_token}/{method}"
        
        data = {
            'chat_id': chat_id,
        }
        
        if topic_id and topic_id.strip():
            data['message_thread_id'] = topic_id.strip()

        if send_mode == "message":
            data['text'] = caption
            try:
                print(f"[Telegram] Sending text message...")
                response = requests.post(url, data=data)
                if response.status_code == 200:
                    print(f"[Telegram] ✅ Message successfully sent!")
                else:
                    print(f"[Telegram Error] Failed to send message: {response.text}")
            except Exception as e:
                print(f"[Telegram Error] Critical error: {e}")
            return ()

        if not file_path or not os.path.exists(file_path):
            print(f"[Telegram Error] File not found or empty path: {file_path}")
            return ()

        data['caption'] = caption
        file_field_name = send_mode

        try:
            with open(file_path, 'rb') as media_file:
                files = {file_field_name: media_file}
                
                print(f"[Telegram] Sending file as '{send_mode}': {file_path} ...")
                response = requests.post(url, data=data, files=files)
                
                if response.status_code == 200:
                    print(f"[Telegram] ✅ Successfully sent!")
                else:
                    print(f"[Telegram Error] Failed to send: {response.text}")
        except Exception as e:
            print(f"[Telegram Error] Critical error: {e}")
            
        return ()


# ==========================================
# 7. TAG READER / VIEWER NODE
# ==========================================
class ReadAudioTags:
    """Reads metadata and cover art from any supported audio file and displays them in UI/Console."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("title", "artist", "album", "genre", "year", "publisher", "copyright", "composer", "album_artist", "bpm", "initial_key", "isrc", "encoder", "track_number", "comment", "lyrics", "license", "cover_image")
    FUNCTION = "read_tags"
    OUTPUT_NODE = True
    CATEGORY = "Audio Tagger"

    def read_tags(self, file_path):
        title = artist = album = genre = year = publisher = copyright_val = composer = album_artist = bpm = initial_key = isrc = encoder = track_number = comment = lyrics = license_val = ""
        cover_tensor = None
        ui_images = []

        if not file_path or not os.path.exists(file_path):
            print(f"[Tag Reader Error] File not found: {file_path}")
            cover_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return {"ui": {}, "result": (title, artist, album, genre, year, publisher, copyright_val, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, comment, lyrics, license_val, cover_tensor)}

        try:
            audio = File(file_path)
            if audio is None or audio.tags is None:
                print(f"[Tag Reader Error] No tags found or unsupported format: {file_path}")
                cover_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
                return {"ui": {}, "result": (title, artist, album, genre, year, publisher, copyright_val, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, comment, lyrics, license_val, cover_tensor)}

            tags = audio.tags
            cover_bytes = None

            # 1. ID3 Tags (MP3, AIFF)
            if isinstance(audio, (MP3, AIFF)) or (hasattr(audio, 'tags') and isinstance(audio.tags, ID3)):
                if 'TIT2' in tags: title = str(tags['TIT2'].text[0])
                if 'TPE1' in tags: artist = str(tags['TPE1'].text[0])
                if 'TALB' in tags: album = str(tags['TALB'].text[0])
                if 'TCON' in tags: genre = str(tags['TCON'].text[0])
                if 'TDRC' in tags: year = str(tags['TDRC'].text[0])
                if 'TPUB' in tags: publisher = str(tags['TPUB'].text[0])
                if 'TCOP' in tags: copyright_val = str(tags['TCOP'].text[0])
                if 'TCOM' in tags: composer = str(tags['TCOM'].text[0])
                if 'TPE2' in tags: album_artist = str(tags['TPE2'].text[0])
                if 'TBPM' in tags: bpm = str(tags['TBPM'].text[0])
                if 'TKEY' in tags: initial_key = str(tags['TKEY'].text[0])
                if 'TSRC' in tags: isrc = str(tags['TSRC'].text[0])
                if 'TSSE' in tags: encoder = str(tags['TSSE'].text[0])
                if 'TRCK' in tags: track_number = str(tags['TRCK'].text[0])
                
                for key in tags:
                    if key.startswith('TXXX'):
                        frame = tags[key]
                        if frame.desc.lower() == 'license':
                            license_val = str(frame.text[0])
                    if key.startswith('COMM'):
                        comment = str(tags[key].text[0])
                    if key.startswith('USLT'):
                        lyrics = str(tags[key].text)
                    if key.startswith('APIC'):
                        cover_bytes = tags[key].data

            # 2. Vorbis Comments (FLAC)
            elif isinstance(audio, FLAC):
                title = tags.get("TITLE", [""])[0]
                artist = tags.get("ARTIST", [""])[0]
                album = tags.get("ALBUM", [""])[0]
                genre = tags.get("GENRE", [""])[0]
                year = tags.get("DATE", [""])[0]
                publisher = tags.get("PUBLISHER", [""])[0]
                copyright_val = tags.get("COPYRIGHT", [""])[0]
                composer = tags.get("COMPOSER", [""])[0]
                album_artist = tags.get("ALBUMARTIST", [""])[0]
                bpm = tags.get("BPM", [""])[0]
                initial_key = tags.get("INITIALKEY", [""])[0]
                isrc = tags.get("ISRC", [""])[0]
                encoder = tags.get("ENCODER", [""])[0]
                track_number = tags.get("TRACKNUMBER", [""])[0]
                license_val = tags.get("LICENSE", [""])[0]
                comment = tags.get("COMMENT", [""])[0]
                lyrics = tags.get("LYRICS", [""])[0]

                if audio.pictures:
                    cover_bytes = audio.pictures[0].data

            # 2b. Vorbis Comments (OGG)
            elif isinstance(audio, OggVorbis):
                title = tags.get("TITLE", [""])[0]
                artist = tags.get("ARTIST", [""])[0]
                album = tags.get("ALBUM", [""])[0]
                genre = tags.get("GENRE", [""])[0]
                year = tags.get("DATE", [""])[0]
                publisher = tags.get("PUBLISHER", [""])[0]
                copyright_val = tags.get("COPYRIGHT", [""])[0]
                composer = tags.get("COMPOSER", [""])[0]
                album_artist = tags.get("ALBUMARTIST", [""])[0]
                bpm = tags.get("BPM", [""])[0]
                initial_key = tags.get("INITIALKEY", [""])[0]
                isrc = tags.get("ISRC", [""])[0]
                encoder = tags.get("ENCODER", [""])[0]
                track_number = tags.get("TRACKNUMBER", [""])[0]
                license_val = tags.get("LICENSE", [""])[0]
                comment = tags.get("COMMENT", [""])[0]
                lyrics = tags.get("LYRICS", [""])[0]

                if "metadata_block_picture" in tags:
                    try:
                        pic_data = base64.b64decode(tags["metadata_block_picture"][0])
                        pic = Picture(pic_data)
                        cover_bytes = pic.data
                    except Exception:
                        pass

            # 3. MP4 Atoms (M4A)
            elif isinstance(audio, MP4):
                title = tags.get("©nam", [""])[0]
                artist = tags.get("©ART", [""])[0]
                album = tags.get("©alb", [""])[0]
                genre = tags.get("©gen", [""])[0]
                year = tags.get("©day", [""])[0]
                publisher = tags.get("©pub", [""])[0]
                copyright_val = tags.get("©cpy", [""])[0]
                composer = tags.get("©wrt", [""])[0]
                album_artist = tags.get("aART", [""])[0]
                bpm = str(tags.get("tmpo", [""])[0]) if "tmpo" in tags else ""
                initial_key = tags.get("----:com.apple.iTunes:INITIALKEY", [b""])[0].decode("utf-8", "ignore") if "----:com.apple.iTunes:INITIALKEY" in tags else ""
                isrc = tags.get("©isr", [""])[0]
                encoder = tags.get("©too", [""])[0]
                
                if "trkn" in tags and tags["trkn"]:
                    track_number = str(tags["trkn"][0][0])

                raw_comment = tags.get("©cmt", [""])[0]
                if "License:" in raw_comment and " | " in raw_comment:
                    parts = raw_comment.split(" | ", 1)
                    license_val = parts[0].replace("License: ", "").strip()
                    comment = parts[1].strip()
                else:
                    comment = raw_comment
                
                lyrics = tags.get("©lyr", [""])[0]

                if "covr" in tags and tags["covr"]:
                    cover_bytes = bytes(tags["covr"][0])

            if cover_bytes:
                cover_tensor = jpeg_bytes_to_tensor(cover_bytes)
            
            if cover_tensor is None:
                cover_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            else:
                temp_dir = folder_paths.get_temp_directory()
                rand_str = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
                filename = f"tag_reader_cover_{rand_str}.png"
                image_np = (cover_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)
                os.makedirs(temp_dir, exist_ok=True)
                img.save(os.path.join(temp_dir, filename))
                ui_images.append({
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp"
                })

            # Print a neat summary to the Python console
            print("=========================================")
            print(f"[Tag Reader] Read Audio Tags for: {os.path.basename(file_path)}")
            print(f"  - Title:        {title}")
            print(f"  - Artist:       {artist}")
            print(f"  - Album:        {album}")
            print(f"  - Genre:        {genre}")
            print(f"  - Year:         {year}")
            print(f"  - Publisher:    {publisher}")
            print(f"  - Copyright:    {copyright_val}")
            print(f"  - Composer:     {composer}")
            print(f"  - Album Artist: {album_artist}")
            print(f"  - BPM:          {bpm}")
            print(f"  - Initial Key:  {initial_key}")
            print(f"  - ISRC:         {isrc}")
            print(f"  - Encoder:      {encoder}")
            print(f"  - Track Number: {track_number}")
            print(f"  - License:      {license_val}")
            print(f"  - Comment:      {comment}")
            print(f"  - Lyrics:       {'Yes (' + str(len(lyrics)) + ' chars)' if lyrics else 'None'}")
            print("=========================================")

        except Exception as e:
            print(f"[Tag Reader Error] Failed to read file {file_path}: {e}")
            cover_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return {
            "ui": {"images": ui_images},
            "result": (title, artist, album, genre, year, publisher, copyright_val, composer, album_artist, bpm, initial_key, isrc, encoder, track_number, comment, lyrics, license_val, cover_tensor)
        }


# ==========================================
# 8. INSPECT ALL TAGS NODE
# ==========================================
class ShowAudioTags:
    """Dumps all metadata tags found in an audio file into a single formatted text block."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags_summary",)
    FUNCTION = "inspect_tags"
    OUTPUT_NODE = False
    CATEGORY = "Audio Tagger"

    def inspect_tags(self, file_path):
        summary_lines = []
        
        if not file_path or not os.path.exists(file_path):
            return (f"File not found or empty path: {file_path}",)

        try:
            audio = File(file_path)
            if audio is None or audio.tags is None:
                return ("No tags found or unsupported format.",)

            summary_lines.append(f"=== Audio File Inspector ===")
            summary_lines.append(f"File: {os.path.basename(file_path)}")
            summary_lines.append(f"Format: {type(audio).__name__}")
            summary_lines.append("-" * 35)

            tags = audio.tags
            if hasattr(tags, 'items'):
                for k, v in tags.items():
                    val_str = str(v)
                    if hasattr(v, 'text'):
                        if isinstance(v.text, str):
                            val_str = v.text
                        elif isinstance(v.text, (list, tuple)):
                            val_str = ", ".join([str(t) for t in v.text])
                        else:
                            val_str = str(v.text)
                    elif isinstance(v, list):
                        val_str = ", ".join([str(item) for item in v])
                    
                    summary_lines.append(f"[{k}] {val_str}")
            else:
                summary_lines.append(str(tags))

            if isinstance(audio, FLAC) and audio.pictures:
                summary_lines.append(f"\n[Cover Art] Yes ({len(audio.pictures)} picture(s) embedded)")
            elif hasattr(tags, 'getall') and tags.getall('APIC'):
                summary_lines.append(f"\n[Cover Art] Yes (ID3 APIC embedded)")
            elif 'covr' in tags:
                summary_lines.append(f"\n[Cover Art] Yes (MP4 Cover embedded)")

        except Exception as e:
            summary_lines.append(f"Error reading tags: {str(e)}")

        full_summary = "\n".join(summary_lines)
        return (full_summary,)


# ==========================================
# NODE REGISTRATION
# ==========================================
NODE_CLASS_MAPPINGS = {
    "SaveMP3WithTags": SaveMP3WithTags,
    "SaveFLACWithTags": SaveFLACWithTags,
    "SaveM4AWithTags": SaveM4AWithTags,
    "SaveOGGWithTags": SaveOGGWithTags,
    "SaveAIFFWithTags": SaveAIFFWithTags,
    "SendAudioToTelegram": SendAudioToTelegram,
    "ReadAudioTags": ReadAudioTags,
    "ShowAudioTags": ShowAudioTags
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveMP3WithTags": "Save MP3 with Tags 🎵",
    "SaveFLACWithTags": "Save FLAC with Tags 🎶",
    "SaveM4AWithTags": "Save M4A with Tags 🍏",
    "SaveOGGWithTags": "Save OGG with Tags 🌐",
    "SaveAIFFWithTags": "Save AIFF with Tags 💿",
    "SendAudioToTelegram": "Send Audio to Telegram ✈️",
    "ReadAudioTags": "Read Audio Tags 🔍",
    "ShowAudioTags": "Show Audio Tags 📋"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']