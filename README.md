![DSF.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/VhKJwA7Tysql8UyvoQWiM.png)

# **IndoorOutdoorNet**

> **IndoorOutdoorNet** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images as either **Indoor** or **Outdoor** using the **SiglipForImageClassification** architecture.

```py
Classification Report:
              precision    recall  f1-score   support

      Indoor     0.9661    0.9554    0.9607      9999
     Outdoor     0.9559    0.9665    0.9612      9999

    accuracy                         0.9609     19998
   macro avg     0.9610    0.9609    0.9609     19998
weighted avg     0.9610    0.9609    0.9609     19998
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/wLvX04YPoU2OsDjKBDKXU.png)


---

The model categorizes images into 2 environment-related classes:

```
    Class 0: "Indoor"
    Class 1: "Outdoor"
```

---

## **Install dependencies**

```python
!pip install -q transformers torch pillow gradio
```

---

## **Inference Code**

```python
import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/IndoorOutdoorNet"  # Updated model name
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def classify_environment_image(image):
    """Predicts whether an image is Indoor or Outdoor."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Indoor", "1": "Outdoor"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=classify_environment_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="IndoorOutdoorNet",
    description="Upload an image to classify it as Indoor or Outdoor."
)

if __name__ == "__main__":
    iface.launch()
```

---

## **Intended Use:**

The **IndoorOutdoorNet** model is designed to classify images into indoor or outdoor environments. Potential use cases include:

- **Smart Cameras:** Detect indoor/outdoor context to adjust settings.
- **Dataset Curation:** Automatically filter image datasets by setting.
- **Robotics & Drones:** Environment-aware navigation logic.
- **Content Filtering:** Moderate or tag environment context in image platforms. 
