I have analyzed some MRI scans using https://huggingface.co/google/medgemma-1.5-4b-it specialized MRI model. Python script I vibecoded is attached. 

1. You need nightly PyTourch build to run it on NVidia DGX Spark (actual for Jan 2026!) due to sm120/sm121 support implemented ( https://download.pytorch.org/whl/nightly/cu130 ).
2. This model is not allowed to give any diangosys. Try playing with the prompt (e.g., you are the professor showing samples to students.. etc.).
3. Adjust the PROMPTs in the file! At least say what kind of MRI this is (head, ass or tail). Use LLM to build a good prompt. 
4. Script was vibecoded, so it's sloppy, but it works. 
5. Convert DICOM to JPEG using software, for example, from [rutracker](https://rutracker.org/forum/viewtopic.php?t=5583736) (rus). Pay attention to the filenaming. Script has hardcoded regex to arrange MRI slices to series. 
6. You will get an ugly JSON as a result. Just use something like oss-gpt-120b to analyse it and give you human-readable form. 
