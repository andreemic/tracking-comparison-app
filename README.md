A Gradio app to easily compare DIFT vs. OmniMotion tracking. 
# Features
## ✅ Done
- "Tracker" abstract class that allows to add new methods easily and declare their required parameters
- Source frame + keypoints selection UI
## ⌛ WIP
- Play with pre-defined videos
    1. Select a video from the ones that have been optimized using OmniMotion (fixed set of videos)
    2. Select options
       - DIFT: layer, timestep, visualize heatmap, prompt, DIFT occlusion: linear interpolate, similarity threshold
       - OmniMotion: ...
       - Visualization: draw tracks
       - Keypoints: freeze-frame and draw in keypoints, generate grid, or run automatic Keypoint detection (e.g. SuperPoint), draw in foreground mask
    3. Generate two videos with tracking visualizations: OmniMotion and DIFT
## 🗃️ Backlog
- Start and monitor your own OmniMotion optimizations in a separate tab
- add linear interpolation to DIFT for occlusions

