import torchaudio as ta
import torch
import sounddevice as sd
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cpu") # [cpu, cuda, gpu]
text = "Welcome to the world of streaming text-to-speech!"

# Basic streaming
# audio_chunks = []
# for audio_chunk, metrics in model.generate_stream(text):
#     audio_chunks.append(audio_chunk)
#     # You can play audio_chunk immediately here for real-time playback
#     print(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}" if metrics.rtf else f"Chunk {metrics.chunk_count}")


# Basic streaming with live audio playing
sr = model.sr 
with sd.OutputStream(samplerate=sr, channels=1, dtype='float32') as stream:

    audio_chunks = []
    for audio_chunk, metrics in model.generate_stream(text):
        # audio_chunk: shape [1, N] torch tensor
        chunk_np = audio_chunk.squeeze(0).cpu().numpy().astype('float32')
        # 🎧 PLAY INSTANTLY
        stream.write(chunk_np)
        audio_chunks.append(audio_chunk)
        print(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}"
              if metrics.rtf else f"Chunk {metrics.chunk_count}")
        
# Combine all chunks into final audio
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("streaming_output.wav", final_audio, model.sr)