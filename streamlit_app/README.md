
# Streamlit Demo of App

app and appv2 are python files that run an interface where user can enter text and get the classification.

Couple of libraries that need to be installed

```
pip install pandas streamlit transformers peft
```

Download the trained model from
https://drive.google.com/drive/folders/1-PffGEcmMqaFLYel0Myo8W0AAKHtitPN?usp=sharing
Then replace the folder "trained_model" with the downloaded folder

To run the version with only bare model,

```
streamlit run app.py
```


To run the version with both bare model and fine tuned,

```
streamlit run appv2.py
```

