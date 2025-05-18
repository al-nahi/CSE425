#Source Code

# Importing necessary libraries
!pip install torch torchvision datasets transformers scikit-learn matplotlib seaborn
!pip install datasets --upgrade


# Load dataset
from datasets import load_dataset

ds = load_dataset("mrm8488/fake-news", split='train')

texts = [item['text'] for item in ds]


# Checking some samples of the the dataset
for i in range(10): print(texts[i])


# Subword Tokenization using BERT and BERT embedding
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def get_bert_embeddings(texts, batch_size=32):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings.append(cls_embeddings)
    return torch.cat(embeddings).numpy()

embeddings = get_bert_embeddings(texts[:2500])  # limit to 2500 for demo


# Clusterng using K-means
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
labels = kmeans.fit_predict(embeddings)


# Evaluation Metrics
from sklearn.metrics import silhouette_score, davies_bouldin_score

print("Silhouette Score:", silhouette_score(embeddings, labels))
print("Davies-Bouldin Index:", davies_bouldin_score(embeddings, labels))


# Visualization of the Clustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_embeddings[:,0], y=tsne_embeddings[:,1], hue=labels, palette='bright')
plt.title("t-SNE Visualization of BERT Embeddings Clusters")
plt.show()
