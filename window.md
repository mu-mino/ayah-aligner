Datenstrukturen                                                                                                                                
  - VideoInfo — Metadaten (Auflösung, FPS, Frame-Anzahl)                                                                                         
  - FrameFeature — Pro-Frame-Metriken: Helligkeit + MSE (ohne Kreiserkennung)                                                                    
  - FrameWindow — Ein erkanntes Fenster mit start_sec, end_sec, peak_frame_idx, peak_sec                                                         
                                                                                                                                                 
Video-Zugriff                                             
- run_ffprobe(path) → VideoInfo                                                                                                                
- get_frame(info, frame_idx) → Graustufenbild             
- stream_features(info) → Helligkeit + MSE per Frame (ohne Kreisdetektion)                                                                     
- get_features(info) → gecachte Variante                                                                                                       
                                                                                                                                                
Segment-Erkennung                                                                                                                              
- find_segments(features) — helle Fenster via Helligkeitsschwelle + MSE-Sprünge                                                                
- filter_segments(segments, fps) — entfernt zu kurze Fenster (< ~2 Sekunden)                                                                   
- pick_peak(features, start, end) → repräsentativster Frame-Index           
                                                                                                                                                
Öffentliche API                                                                                                                                
- extract_windows(video_path) → List[FrameWindow]