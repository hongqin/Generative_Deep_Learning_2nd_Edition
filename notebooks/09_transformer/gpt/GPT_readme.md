# GPT Notebook Overview

This document summarizes the `gpt.ipynb` notebook in `notebooks/09_transformer/gpt/`, which walks through training a miniature GPT-style language model on the wine reviews dataset.

## 1. Notebook Goals and Setup
- Introduces the objective of training a GPT model on structured wine review text, referencing the original Keras tutorial that inspired the workflow.
- Imports TensorFlow/Keras, NumPy, and utility modules while enabling IPython autoreload for iterative experimentation.
- Establishes hyperparameters such as vocabulary size, sequence length, embedding dimensions, transformer head count, batch size, and training epochs.

## 2. Data Loading and Cleaning
- Loads the full Wine Enthusiast dataset from the project data directory and filters out entries with missing country, province, variety, or description fields.
- Converts each record into a templated prompt (`"wine review : {country} : {province} : {variety} : {description}"`) that supplies structured context followed by free-form text.

## 3. Tokenization Pipeline
- Applies a punctuation-padding helper to ensure punctuation marks become standalone tokens before lowercasing and vectorization.
- Builds a `tf.data.Dataset` from the preprocessed strings and adapts a `TextVectorization` layer to learn the vocabulary and map text to integer sequences of fixed length.
- Prepares input-target pairs by shifting sequences one token ahead so that the model learns to predict the next word in the sequence.

## 4. Model Architecture
- Defines a causal attention mask to enforce left-to-right autoregressive conditioning within the transformer block. The helper builds a lower-triangular matrix by comparing destination and source indices with `tf.range`, reshapes it to `[1, seq_len, seq_len]`, and tiles it to match the runtime batch size so that each token only attends to itself and past tokens during self-attention.
- Details the cross-attention-style computation performed inside `layers.MultiHeadAttention`. For the self-attention used here, the same `[batch, seq_len, embed_dim]` tensor provides the queries **Q**, keys **K**, and values **V**. Each head first applies its own learned linear projections that reshape the tensor to `[batch, num_heads, seq_len, key_dim]`—`seq_len` is preserved because attention does not alter time steps, while the innermost dimension becomes the per-head `key_dim` (set to `KEY_DIM = 256` in the notebook). Attention weights are computed as `softmax((Q @ K^T) / sqrt(key_dim) + mask)`, where `Q @ K^T` produces a `[batch, num_heads, seq_len, seq_len]` score matrix whose `(b, h, i, j)` entry compares token `i`’s query against token `j`’s key. The resulting weights linearly combine the value vectors so that the attended output for position `i` is `sum_j weight[b, h, i, j] * V[b, h, j, :]`, yielding `[batch, num_heads, seq_len, key_dim]` tensors. Concatenating the heads therefore gives `[batch, seq_len, num_heads * key_dim]` (i.e., `[batch, 80, 512]` with two heads), which the layer’s output projection maps back to the model’s `embed_dim` of 256.
- Implements a custom `TransformerBlock` layer composed of multi-head self-attention, residual connections, layer normalization, and a two-layer feed-forward network.
- Creates a `TokenAndPositionEmbedding` layer that sums learned token embeddings with learned positional embeddings for each time step.
- Assembles the GPT decoder by stacking the embedding layer, a single transformer block, and a softmax projection over the vocabulary. The model outputs both token logits and attention weights for inspection during generation.

## 5. Training Loop and Generation Utilities
- Configures callbacks for checkpointing, TensorBoard logging, and on-the-fly text sampling via a custom `TextGenerator` callback that performs temperature-controlled sampling.
- Trains the model on the prepared dataset for the specified number of epochs and saves the resulting weights for later reuse.
- Provides helper routines to visualize attention-weighted prompts and to generate sample wine reviews conditioned on different country prompts.

## 6. Key Takeaways
- The notebook demonstrates how a compact GPT architecture can be trained from scratch using Keras primitives.
- Careful preprocessing (structured prompts, punctuation handling, sequence shifting) is essential for effective autoregressive modeling.
- Exposing attention scores and integrating interactive callbacks helps interpret and monitor generation quality during and after training.
