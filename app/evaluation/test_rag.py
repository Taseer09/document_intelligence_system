from ragas import evaluate
from datasets import Dataset

data = {
    "question": [
        "What is the main topic of the document?"
    ],
    "answer": [
        "The document discusses..."
    ],
    "contexts": [
        ["text chunk containing the answer"]
    ]
}

dataset = Dataset.from_dict(data)

results = evaluate(dataset)

print(results)