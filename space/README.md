---
title: Laya · who or what causes an AI risk
emoji: 🦉
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Label an AI risk along the MIT Causal Taxonomy
models:
- MicheleStingo/laya-kindynos-causal
datasets:
- MicheleStingo/kindynos-causal-risks
tags:
- ai-risk
- ai-safety
- risk-assessment
- mit-ai-risk-repository
---

# Laya · who or what causes an AI risk

Describe an AI risk, and [laya-kindynos-causal](https://huggingface.co/MicheleStingo/laya-kindynos-causal)
labels it along the three axes of the [MIT Causal Taxonomy of AI Risks](https://airisk.mit.edu/):

- **entity**: who or what causes it, an AI system, people, or something else;
- **intent**: whether the harm is an expected or an unexpected outcome of pursuing a goal;
- **timing**: whether it occurs before or after the AI system is deployed.

Each value comes with a calibrated probability. The model reads English and Italian and runs on a CPU. It is
[Laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) fine-tuned for
[Kindynos](https://github.com/6MicheleStingo9/kindynos), an open-source tool that assesses the risks of an AI
system from a questionnaire. The examples are risks that Kindynos wrote, from its test set.

The optional subdomain of the MIT Domain Taxonomy goes with the text, as in Kindynos. The text you write is
not stored.

## From code

The demo is also an API:

```python
from gradio_client import Client

client = Client("MicheleStingo/laya-kindynos-causal")
_, labels = client.predict(
    "4.3",  # subdomain of the MIT Domain Taxonomy, or "" if unknown
    "Exploitation for fraud, scams, or targeted manipulation",
    "The system could be misused for generating convincing phishing messages or targeted scams.",
    api_name="/label",
)
print(labels["entity"]["value"], labels["entity"]["probabilities"])
```

To label many risks, load the model itself: see the
[model card](https://huggingface.co/MicheleStingo/laya-kindynos-causal).

## Credits

The taxonomies come from the MIT AI Risk Repository (CC BY 4.0): Slattery, P., Saeri, A. K., Grundy, E. A. C.,
Graham, J., Noetel, M., Uuk, R., Dao, J., Pour, S., Casper, S., & Thompson, N. (2024). *The AI Risk Repository:
A Comprehensive Meta-Review, Database, and Taxonomy of Risks from Artificial Intelligence.*
https://doi.org/10.48550/arXiv.2408.12622

The source of this Space is in the [Kindynos repository](https://github.com/6MicheleStingo9/kindynos/tree/main/space).
