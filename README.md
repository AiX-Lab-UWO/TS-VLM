# TS-VLM: Text-Guided SoftSort Pooling for Vision-Language Models in Multi-View Driving Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2505.12670-b31b1b.svg)](https://arxiv.org/abs/2505.12670)

## Abstract

Vision-Language Models (VLMs) have shown remarkable potential in advancing autonomous driving by leveraging multi-modal fusion in order to enhance scene perception, reasoning, and decision-making. Despite their potential, existing models suffer from computational overhead and inefficient integration of multi-view sensor data that make them impractical for real-time deployment in safety-critical autonomous driving applications. To address these shortcomings, this paper is devoted to designing a lightweight VLM called TS-VLM, which incorporates a novel Text-Guided SoftSort Pooling (TGSSP) module. By resorting to semantics of the input queries, TGSSP ranks and fuses visual features from multiple views, enabling dynamic and query-aware multi-view aggregation without reliance on costly attention mechanisms. This design ensures the query-adaptive prioritization of semantically related views, which leads to improved contextual accuracy in multi-view reasoning for autonomous driving. Extensive evaluations on the DriveLM benchmark demonstrate that, on the one hand, TS-VLM outperforms state-of-the-art models with a BLEU-4 score of 56.82, METEOR of 41.91, ROUGE-L of 74.64, and CIDEr of 3.39. On the other hand, TS-VLM reduces computational cost by up to 90%, where the smallest version contains only 20.1 million parameters, making it more practical for real-time deployment in autonomous vehicles. 

## Usage

### Installation
```bash
git clone https://github.com/AiX-Lab-UWO/TS-VLM.git
conda env create -f environment.yaml
conda activate TSVLM
```

### Data Preparation
* Step 1: Download the dataset from the below link:
[data](https://drive.google.com/file/d/10Fp9_cZJO9R1RYJxUTEeJHB_Lk0UEC_3/view?usp=sharing)

* Step 2: Organize the downloaded files in the following way.
```bash
├─ data
├─ train.py
├─ eval.py
├─ TSVLM_dataset.py
├─ TSVLM.py
...
```

### Train
The training results will be stored at `./results`. For additional training hyperparameter options, please refer to the full argument list in train.py.
```bash
python train.py --lm T5-Tiny
```

### Evaluation
To evaluate the trained model, use the following command. For additional training hyperparameter options, please refer to the full argument list in eval.py.
```bash
python eval.py --model-name your_modelname
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.