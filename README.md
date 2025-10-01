[![Demo](https://img.shields.io/badge/Demo-Colab-orange)](#demo)
# Donut Model Fine-tuning for Document Understanding

A comprehensive repository for fine-tuning the Donut model for document image classification and parsing tasks. This project provides optimized training pipelines using Hugging Face Transformers with custom enhancements for accurate loss calculation.

## Features

- **🚀 Optimized Training:** Leverages Hugging Face Trainer for distributed and optimized training
- **📊 Multiple Tasks:** Supports both document classification and document parsing
- **🔧 Custom Fixes:** Includes a patched DonutModel class for accurate loss calculation
- **🆓 Colab Compatible:** All notebooks run on Google Colab free tier

## What is Donut?

Donut (Document Understanding Transformer) is a novel transformer-based model that doesn't require OCR for document understanding. It directly processes document images and generates structured outputs.
![donut-arch](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/donut_architecture.jpg)


## Resources

| notebook | task | dataset | finetuned model | tutorial |
|---|---|---|---|---|
| [doc-parsing-colab](https://colab.research.google.com/drive/1o4FzEZn4GZWxgbdiRgUmAx1bDWnOQHW2?usp=sharing) | information extraction | [SROIE-document-parsing](https://huggingface.co/datasets/hf-tuner/SROIE-document-parsing) | [donut-base-finetuned-sroie](https://huggingface.co/hf-tuner/donut-base-finetuned-sroie) | [youtube-tutorial](https://www.youtube.com/watch?v=Ucu39UY3Vtg)
| [doc-classification-colab](https://colab.research.google.com/drive/18ApbtvvMtWl1DWJR_9D1yyrHBxzZZ_AA?usp=sharing) | document classification | [rvl-cdip-document-classification](https://huggingface.co/datasets/hf-tuner/rvl-cdip-document-classification) | [donut-base-finetuned-rvl-cdip](https://huggingface.co/hf-tuner/donut-base-finetuned-rvl-cdip) | [youtube-tutorial](https://www.youtube.com/watch?v=a2CH3LCpD7I)
| [doc-vqa-colab](https://colab.research.google.com/drive/1O6skrn0IhoSv4dfEyJYzBLGJWJ_F-pVq?usp=sharing) | visual question answering | [docvqa-10k-donut](https://huggingface.co/datasets/hf-tuner/hf-tuner/docvqa-10k-donut) | [donut-base-finetuned-docvqa](https://huggingface.co/hf-tuner/donut-base-finetuned-docvqa) | [youtube-tutorial](https://www.youtube.com/watch?v=Mmu3dHq0zV4)


## Acknowledgments
- The [original Donut repo](https://github.com/clovaai/donut)
- Hugging Face for the transformers and datasets library
- Google Colab for providing free GPU resources

## Citation

### Original Donut Paper
```bibtex
@inproceedings{kim2022donut,
  title     = {OCR-Free Document Understanding Transformer},
  author    = {Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```
