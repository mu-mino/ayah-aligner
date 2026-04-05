  Datenstrukturen                                                                                                                                
  - TranscriptSegment — ein WhisperX-Segment: start, end, text (absolut in Videosekunden)                                                        
  - ChunkTranscription — das Ergebnis für ein Häppchen: das zugehörige FrameWindow + Segmente + Volltext                                         
                                                                                                                                                 
Kernfunktionen                                                                                                                                 
  - load_model(device, model_name, compute_type) — lädt WhisperX einmalig (large-v2, Arabisch); erkennt automatisch CUDA/CPU und wechselt bei CPU
    auf int8                                                                                                                                      
  - _extract_audio_chunk(video_path, start_sec, end_sec, tmp_path) — ffmpeg: mono WAV, 16 kHz für den genauen Zeitabschnitt                    
  - transcribe_chunk(model, video_path, window) → ChunkTranscription — ein Häppchen transkribieren; WhisperX-relative Zeiten werden auf absolute 
Videosekunden umgerechnet                                                                                                                      
  - transcribe_chunks(video_path, windows, model) → List[ChunkTranscription] — Entry-Point für das Orchestrierungs-Modul                         