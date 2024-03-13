# FAQ
- **Q**: 21/1/2024: Hi there, do you guys have any rough baselines for inference time of the test/validation sets (of course it will be GPU dependant). The full sn-gamestate baseline pipeline takes a very long time to run on my machinen using the provided weights, but I'm not sure what a normal time should be
  - **A**: Hi, for inference time of the sn-gamestate baseline, it should take around 12min per video on GPU, so around 12hours for an entire set. Some part of the baseline are very slow, such as jersey number recognition, and we strongly encourage participants to find faster and more accurate methods! Moreover, do not forget to play around with the batch size of various models (in soccernet.yaml), to ensure maximum GPU usage all the time.