import argparse
import os
import sys
import torch
import gc
from TTS.api import TTS
from rvc_python.infer import RVCInference

# Ensure UTF-8 output for Windows consoles to avoid UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def generate_voice(text, output_path, ref_audio, rvc_model, rvc_index):
    print(f"Generating voice for: {text}")
    
    # 1. TTS (Coqui XTTS)
    # Initialize TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading TTS on {device}...")
    
    # Workaround for PyTorch 2.6+ default weights_only=True if needed, 
    # but usually handled by the specific env versions.
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    # Generate TTS Audio
    temp_tts_path = output_path.replace(".wav", "_tts.wav")
    
    # XTTS v2 supports Arabic (ar)
    tts.tts_to_file(text=text, speaker_wav=ref_audio, language="ar", file_path=temp_tts_path)
    
    # Optimization: Unload TTS to free VRAM for RVC
    print("Releasing TTS memory...")
    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. RVC
    print("Applying RVC...")
    try:
        # Initialize the new class
        rvc = RVCInference(device=device)
        
        # Load the model
        rvc.load_model(rvc_model)
        
        # Set your parameters (rmvpe, pitch=0)
        rvc.set_params(f0method="rmvpe", f0up_key=0, index_path=rvc_index)
        
        # Run inference
        rvc.infer_file(temp_tts_path, output_path)
        
        print(f"Success! Saved to {output_path}")
    except Exception as e:
        print(f"RVC Failed: {e}")
        # Fallback to TTS output if RVC fails
        if os.path.exists(temp_tts_path):
            os.rename(temp_tts_path, output_path)
    
    # Cleanup
    if os.path.exists(temp_tts_path) and os.path.exists(output_path):
        os.remove(temp_tts_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--rvc_model", required=True)
    parser.add_argument("--rvc_index", required=True)
    
    args = parser.parse_args()
    
    generate_voice(args.text, args.output, args.ref_audio, args.rvc_model, args.rvc_index)