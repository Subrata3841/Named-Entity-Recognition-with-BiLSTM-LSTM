
# Named Entity Recognition (NER) using BiLSTM-LSTM in Keras

This project implements Named Entity Recognition (NER) using a combination of Bidirectional LSTM and LSTM layers with Keras. It also includes visualization using spaCy's pretrained model to display named entities in text.

---
## 🧠 Model Architecture

- **Embedding Layer**: Converts words into dense vector representations.
- **Bidirectional LSTM**: Captures information from both past and future states.
- **LSTM Layer**: Learns sequence dependencies.
- **TimeDistributed Dense Layer**: Maps the output to NER tag space with softmax activation.

---

## 📊 Dataset

The model is trained on the popular `ner_dataset.csv`, which includes:

- **Word**: The input token.
- **POS**: Part of Speech tag.
- **Tag**: The named entity tag for each word (e.g., B-PER, I-LOC, O, etc.)

> Ensure the dataset file `ner_dataset.csv` is present in your project root.

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- NumPy & Pandas
- Scikit-learn
- spaCy (for visualization)
- Google Colab (for training environment)

---

## 🛠️ How to Run

1. **Upload Dataset**  
   Upload `ner_dataset.csv` using:
   python
   ```
   from google.colab import files
   uploaded = files.upload()```

3. **Install Dependencies**
   Install the required packages in Colab:

   ```bash
   pip install tensorflow keras spacy
   python -m spacy download en_core_web_sm
   ```

4. **Run Notebook Cells**
   Follow the notebook cells to:

   * Preprocess the data
   * Build and train the BiLSTM-LSTM model
   * Evaluate the model
   * Visualize named entities using spaCy

5. **Named Entity Visualization**
   Run the following snippet to visualize entities:

   ```python
   import spacy
   from spacy import displacy

   nlp = spacy.load("en_core_web_sm")
   text = "Hi, My name is Subrata Mondal. I am from India. I want to work with Google. Virat Kohli is My Inspiration."
   doc = nlp(text)
   displacy.serve(doc, style="ent")
   ```

## 📈 Training Output

The model is trained for 25 epochs using categorical cross-entropy loss and Adam optimizer.

```python
model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
```

Training and validation loss is plotted for analysis.

---

## 📷 Sample Output

Example sentence:

> "My name is Subrata and I am a Machine Learning Student."

Detected Entities:

* **Subrata** → PERSON
* **Machine Learning** → FIELD
* **Student** → PROFESSION

## 📁 Folder Structure

```
├── ner_dataset.csv
├── Named_Entity_Recognition.ipynb
├── README.md
```
## 🙋‍♂️ Author

**Subrata Mandal**
Connect on [LinkedIn](https://www.linkedin.com/in/subratamondal38/) | GitHub: [@subrata38](https://github.com/subrata38)
