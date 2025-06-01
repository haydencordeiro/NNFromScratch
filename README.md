# 🧠 NNFromScratch - Learn Backpropagation by Building a Neural Network from Scratch  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)

---

## 📚 Table of Contents

- [Overview](#overview)
  - [Objective](#objective)
  - [Motivation](#motivation)
- [Features](#features)
- [Visual Demo](#visual-demo)
- [Built With](#built-with)
- [Disclaimer](#disclaimer)
- [Author](#author)

---

## 🧠 Overview

### Objective
To help beginners understand **how neural networks actually learn**, by building a neural network **from scratch** (no frameworks) to solve the **XOR classification problem**—a classic case of non-linear separability.

### Motivation
Most tutorials abstract away the learning process using libraries. This project strips it all down, exposing how **forward propagation**, **backpropagation**, and **weight updates** actually work—using only raw Python, math, and logic.

---

## 🚀 Features

- Custom implementation of a feedforward neural network  
- XOR gate classification  
- Manual implementation of:
  - Sigmoid activation function and its derivative
  - Forward pass
  - Backpropagation algorithm
  - Weight and bias updates
- Visual explanation of neuron flows and errors (diagrammatic)
- Matplotlib plots saved per iteration for animation

---

## 🎞️ Visual Demo
[![youtube](https://img.youtube.com/vi/PkKaKe0_xCA/0.jpg)](https://youtu.be/PkKaKe0_xCA)


> You can generate your own training animation using the code in the repo.

---

## 🛠️ Built With

- Python
- NumPy
- Matplotlib

---

## ⚠️ Disclaimer

The purpose of this repository is to help you **understand how a neural network learns**, not to serve as a production-grade classifier. For simplicity, we use **Mean Squared Error (MSE)** as the loss function.

While MSE is mathematically easier to follow and great for educational purposes, it’s **not ideal for classification tasks**. In real-world binary classification, **Cross-Entropy Loss** is preferred.

---

