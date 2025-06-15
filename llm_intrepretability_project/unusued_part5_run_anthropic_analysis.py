import argparse
import asyncio
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import json
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from tqdm.auto import tqdm
import random
from typing import List, Tuple, Dict, Any, Optional
import html
import time
import datetime
import logging
import os
import requests # Import requests for Claude API calls
import re # Import re for parsing score

# Suppress matplotlib debug messages
logging.getLogger('matplotlib').setLevel(logging.WARNING)
# Suppress transformers warning about legacy tokenizer
logging.getLogger('transformers.models.llama.tokenization_llama').setLevel(logging.ERROR) # Or WARNING

class SparseAutoencoder(torch.nn.Module):
    """Simple Sparse Autoencoder module."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Encoder maps input activation vector to sparse feature vector
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU() # ReLU enforces non-negativity for sparsity
        )
        # Decoder maps sparse feature vector back to original activation space
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        # x: input activation vector (batch_size, seq_len, input_dim)
        features = self.encoder(x) # features: (batch_size, seq_len, hidden_dim)
        # features are already non-negative due to ReLU

        # Reconstruction maps features back to the input activation space
        reconstruction = self.decoder(features) # reconstruction: (batch_size, seq_len, input_dim)

        return features, reconstruction # Return both the features and the reconstruction

class SAEFeatureAnalyzer:
    """Analyzer for studying features in a Sparse Autoencoder trained on language model activations."""

    def __init__(self, sae_model, tokenizer, base_model, device="cuda", target_layer=16,
                 identification_prompt=None, scoring_prompt=None, claude_api_key=None):
        self.sae_model = sae_model
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.device = device
        self.target_layer = target_layer
        self.sae_model.eval()  # Set to evaluation mode
        self.base_model.eval()

        # Store feature statistics
        self.feature_stats = {}

        # Store prompts
        self.identification_prompt = identification_prompt
        self.scoring_prompt = scoring_prompt

        # Claude API key
        self.claude_api_key = claude_api_key

    # NOTE: API call estimation is now handled dynamically in main based on args

    def find_most_active_features(self, texts, n_features=10, n_samples=1000, batch_size=16,
                                 activation_threshold=10.0, max_activation_threshold=None):
        """
        Find features that have high activations across the dataset, randomly selecting from those above threshold.
        Uses simple efficient batching.

        Args:
            texts: List of text samples
            n_features: Number of features to return
            n_samples: Number of text samples to use for feature discovery
            batch_size: Number of texts to process in each batch
            activation_threshold: Minimum activation value to consider a feature
            max_activation_threshold: Maximum activation value to consider a feature (useful to filter out special tokens)

        Returns:
            List of (feature_idx, max_activation) tuples
        """
        # Ensure n_samples is an integer
        n_samples = int(n_samples)

        # Sample texts if needed
        if len(texts) > n_samples:
            sampled_texts = random.sample(texts, n_samples)
        else:
            sampled_texts = texts

        feature_max_activations = {}
        # Total number of features is the hidden dimension of the SAE
        n_features_total = self.sae_model.encoder[0].out_features

        # Get special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}

        print(f"Scanning {len(sampled_texts)} texts to discover active features (in batches of {batch_size})...")
        print(f"Ignoring special tokens: {special_tokens}")

        # Process in batches
        for i in tqdm(range(0, len(sampled_texts), batch_size), desc="Discovering features"):
            batch_texts = sampled_texts[i:i+batch_size]
            # Filter empty texts
            batch_texts = [text for text in batch_texts if text.strip()]

            if not batch_texts:
                continue

            # Tokenize batch directly - HuggingFace handles padding efficiently
            # Using return_tensors="pt" and moving to device inside the loop
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to(self.device)
            # Token IDs are needed for checking special tokens, so move to cpu *after* tokenizing
            token_ids = inputs["input_ids"].cpu().numpy()


            with torch.no_grad():
                # Process batch through model
                outputs = self.base_model(**inputs, output_hidden_states=True)
                # Access hidden states of the target layer
                hidden_states = outputs.hidden_states[self.target_layer]

                # Process through SAE
                # Ensure hidden_states is float32 for SAE if SAE was trained in float32
                # Although the SAE model is moved to device, ensure input dtype matches expected
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))


                # For each sample in batch
                for b in range(features.shape[0]):
                    # Get valid sequence length (exclude padding)
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()

                    # Create a mask for non-special tokens within the sequence length
                    token_mask = torch.ones(seq_len, dtype=torch.bool, device=self.device)

                    # Convert token IDs to tokens for this sample
                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    # Mark special tokens to ignore
                    for pos, token in enumerate(sample_tokens):
                         # Check both token string and token ID value if known special token IDs exist
                         # For Llama, EOS/BOS are typically 1 and 2. Pad might be EOS.
                         # Better to check decoded token string and potentially ID if standard
                         if token in special_tokens or sample_token_ids[pos] in self.tokenizer.all_special_ids:
                            token_mask[pos] = False

                    # If all tokens are special or seq_len is 0, skip this sample
                    if seq_len == 0 or not torch.any(token_mask):
                        continue

                    # Apply mask to features before finding max - set features for masked tokens to 0
                    masked_features = features[b, :seq_len].clone()
                    masked_features[~token_mask] = 0 # Set features for special tokens to 0

                    # Get max activation per feature over valid (non-special) tokens only
                    # Max should now be 0 for features that only activated on special tokens
                    max_activations, _ = torch.max(masked_features, dim=0)


                    # Update global maxima for non-zero activations that meet threshold
                    non_zero_idxs = torch.where(max_activations > 0)[0] # Consider only features with >0 max activation in this sample
                    for feature_idx in non_zero_idxs:
                         feat_idx = feature_idx.item()
                         activation_val = max_activations[feature_idx].item()

                         if activation_val >= activation_threshold: # Check against minimum threshold
                             if max_activation_threshold is None or activation_val <= max_activation_threshold: # Check against maximum threshold
                                 if feat_idx not in feature_max_activations or activation_val > feature_max_activations[feat_idx]:
                                     feature_max_activations[feat_idx] = activation_val

        # Filter features within the activation threshold range (already done in the loop, but re-collect)
        valid_features = [(idx, val) for idx, val in feature_max_activations.items()]


        print(f"Found {len(valid_features)} features within activation range "
              f"[{activation_threshold}, {max_activation_threshold if max_activation_threshold is not None else 'inf'}] "
              f"that had >0 activation on non-special tokens.")

        # If we have enough features in range, randomly sample from them
        if len(valid_features) >= n_features:
            selected_features = random.sample(valid_features, n_features)
            # Sort by activation value for display purposes
            return sorted(selected_features, key=lambda x: x[1], reverse=True)
        else:
            # Not enough features in range, use all we have that meet the min threshold
            print(f"Warning: Only {len(valid_features)} features found within specified activation range "
                  f"[{activation_threshold}, {max_activation_threshold if max_activation_threshold is not None else 'inf'}] "
                  f"with >0 activation on non-special tokens.")
            if len(valid_features) == 0:
                print("No features found meeting the minimum threshold on non-special tokens. Cannot proceed with analysis.")
                return [] # Return empty list if no features meet criteria
            else:
                # Use all found valid features
                return sorted(valid_features, key=lambda x: x[1], reverse=True)


    def find_highest_activating_examples(self, feature_idx, texts, top_n=20, batch_size=16, window_size=10):
        """
        Find texts with highest token-level activations for a feature.
        Uses simple efficient batching.

        Args:
            feature_idx: The feature to analyze
            texts: List of text samples
            top_n: Number of top examples to return
            batch_size: Number of texts to process in each batch
            window_size: Number of tokens before and after the highest activation to include (max 10)

        Returns:
            List of dicts with text, max_activation, max_position, and max_token
        """
        # Limit window size to maximum of 10 tokens
        window_size = min(window_size, 10)

        results = []

        # Get special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}

        print(f"Scanning for feature {feature_idx} to find top {top_n} examples (batch size {batch_size}, window size {window_size})...")
        # print(f"Ignoring special tokens: {special_tokens}") # Already printed in find_most_active_features

        # Process in batches
        # Store results temporarily and sort at the end to get top_n overall
        temp_results = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Examples for Feature {feature_idx}"):
            batch_texts = texts[i:i+batch_size]
            # Filter empty texts
            batch_texts = [text for text in batch_texts if text.strip()]

            if not batch_texts:
                continue

            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy()


            with torch.no_grad():
                # Process batch through model
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                # Process through SAE
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                # For each sample in batch
                for b in range(features.shape[0]):
                    # Get valid sequence length (exclude padding)
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()

                    if seq_len == 0: continue

                    # Get activations for this feature for the valid sequence length
                    token_activations = features[b, :seq_len, feature_idx]

                    # Only consider if there's non-zero activation on non-special tokens
                    # Create a mask for non-special tokens
                    token_mask = torch.ones_like(token_activations, dtype=torch.bool)

                    # Convert token IDs to tokens for this sample to check against special tokens
                    sample_token_ids = token_ids[b][:seq_len]
                    sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                    for pos, token in enumerate(sample_tokens):
                        if token in special_tokens or sample_token_ids[pos] in self.tokenizer.all_special_ids:
                            token_mask[pos] = False

                    # If all tokens are special or no activation on non-special tokens, skip
                    if not torch.any(token_mask) or torch.max(token_activations[token_mask]) <= 0:
                        continue

                    # Apply mask and find max activation among non-special tokens
                    # Set activations for special tokens to -inf or 0 to ensure max is from non-special
                    masked_activations = token_activations.clone()
                    masked_activations[~token_mask] = -1e9 # Set to a very low value

                    max_activation = torch.max(masked_activations).item()

                    # If max activation among non-special tokens is <= 0 after masking, skip
                    if max_activation <= 0:
                         continue

                    # Get the position of the max activation within the valid sequence length
                    max_position = torch.argmax(masked_activations).item()


                    # Ensure max_position is within valid range (should be if derived from seq_len)
                    if max_position < seq_len:
                        # Get the token at the max activation position
                        max_token_id = token_ids[b, max_position]
                        max_token = self.tokenizer.decode([max_token_id], skip_special_tokens=False) # Decode potentially special token for logging

                        # Calculate window around max activation, limited to max 10 tokens on each side
                        start_pos = max(0, max_position - window_size)
                        end_pos = min(seq_len, max_position + window_size + 1)

                        # Get windowed token ids and decode to text
                        window_token_ids = token_ids[b, start_pos:end_pos]
                        # Decode windowed text *without* skipping special tokens initially, then clean
                        window_text_raw = self.tokenizer.decode(window_token_ids, skip_special_tokens=False)

                        # Clean specific special tokens from the windowed text for presentation
                        window_text_cleaned = window_text_raw.replace(self.tokenizer.bos_token or "<s>", "") \
                                                         .replace(self.tokenizer.eos_token or "</s>", "") \
                                                         .replace(self.tokenizer.pad_token or "", "") \
                                                         .replace(self.tokenizer.cls_token or "", "") \
                                                         .replace(self.tokenizer.sep_token or "", "")
                        window_text_cleaned = window_text_cleaned.strip() # Strip leading/trailing whitespace after removal


                        temp_results.append({
                            "text": batch_texts[b], # Store original full text
                            "windowed_text": window_text_cleaned, # Store cleaned windowed text
                            "max_activation": max_activation,
                            "max_position": int(max_position), # Ensure serializable
                            "max_token": max_token,
                            "window_start": int(start_pos), # Ensure serializable
                            "window_end": int(end_pos) # Ensure serializable
                        })


        # Sort all collected results by max activation and return top_n
        temp_results.sort(key=lambda x: x["max_activation"], reverse=True)
        return temp_results[:top_n]

    def get_token_feature_activations(self, text, feature_idx, window_size=10):
        """Get activations of a specific feature for each token in the text, with window limiting."""
        # Tokenize the text
        # Use padding=True even for single text to get attention mask for seq_len
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)

        # Get tokens for filtering special tokens and sequence length
        seq_len = torch.sum(inputs["attention_mask"][0]).item()
        if seq_len == 0:
             logging.getLogger('debug').warning(f"Input text '{text[:50]}...' tokenized to empty sequence.")
             return [] # Return empty if sequence is empty

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][:seq_len])
        token_ids = inputs["input_ids"][0][:seq_len].cpu().numpy()

        # Identify special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}
        special_ids = set(self.tokenizer.all_special_ids)


        with torch.no_grad():
            # Process through model
            outputs = self.base_model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[self.target_layer]

            # Process through SAE
            features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

            # Extract activations for the specific feature for the valid sequence length
            token_activations = features[0, :seq_len, feature_idx].cpu().numpy()

        # Create a mask for non-special tokens based on token string and ID
        token_mask = np.ones_like(token_activations, dtype=bool)
        for i in range(seq_len):
            if tokens[i] in special_tokens or token_ids[i] in special_ids:
                token_mask[i] = False


        # Apply mask to token activations for finding max (set special token activations to 0)
        masked_activations = token_activations.copy()
        masked_activations[~token_mask] = 0

        # Find max activation token index among non-special tokens.
        # If no non-special token has >0 activation, np.argmax on masked_activations will give index of a 0.
        # Need to check if max is > 0 *after* masking.
        max_activation_idx = np.argmax(masked_activations)
        max_val_after_masking = masked_activations[max_activation_idx]


        # If this is a windowed text (likely short) or if no non-special token activated > 0, use all valid tokens
        if seq_len <= window_size * 2 + 1 or max_val_after_masking <= 0:
            # Return all non-special tokens and their activations
            return [(tokens[i].replace('Ġ', ' ').replace(' ', ' ').replace('<s>', '').replace('</s>', ''), token_activations[i])
                    for i in range(seq_len) if token_mask[i]] # Only include non-special tokens in the result


        # If it's a full text and non-special tokens activated, find the max activation token and window around it
        start_idx = max(0, max_activation_idx - window_size)
        end_idx = min(seq_len, max_activation_idx + window_size + 1)

        # Match tokens with activations within the window, only include non-special tokens
        return [(tokens[i].replace('Ġ', ' ').replace(' ', ' ').replace('<s>', '').replace('</s>', ''), token_activations[i])
                for i in range(start_idx, end_idx) if token_mask[i]]


    def visualize_token_activations(self, text, feature_idx, output_dir=None, window_size=10):
        """
        Create matplotlib visualization of token activations.
        Uses color intensity to show activation strength.

        Args:
            text: Text to visualize (this is expected to be the windowed text snippet)
            feature_idx: Feature index to visualize
            output_dir: Directory to save visualization (if None, uses temp dir)
            window_size: Max number of tokens before and after highest activation to visualize
                         (Note: this function operates on the already windowed text)

        Returns:
            Path to the saved image
        """
        # When visualizing, we are given the windowed text, so just get activations for that text
        # The windowing logic is handled in find_highest_activating_examples and get_token_feature_activations
        # This function just needs activations for the provided text snippet.

        # Get token activations for the provided text snippet
        # We don't re-window here, just get activations for the input text sequence
        # Filter out special tokens from the *input text for visualization*
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(self.device)
        seq_len = torch.sum(inputs["attention_mask"][0]).item()
        if seq_len == 0:
             logging.getLogger('debug').warning(f"Visualization input text '{text[:50]}...' tokenized to empty sequence.")
             tokens_to_viz = []
             activations_to_viz = []
        else:
            tokens_raw = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][:seq_len])
            token_ids = inputs["input_ids"][0][:seq_len].cpu().numpy()
            special_tokens_set = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                                  self.tokenizer.pad_token, self.tokenizer.cls_token,
                                  self.tokenizer.sep_token, "<s>", "</s>"])
            special_ids_set = set(self.tokenizer.all_special_ids)


            with torch.no_grad():
                 outputs = self.base_model(**inputs, output_hidden_states=True)
                 hidden_states = outputs.hidden_states[self.target_layer]
                 features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))
                 all_activations = features[0, :seq_len, feature_idx].cpu().numpy()

            tokens_to_viz = []
            activations_to_viz = []
            for i in range(seq_len):
                 # Only include non-special tokens for visualization display
                 if tokens_raw[i] not in special_tokens_set and token_ids[i] not in special_ids_set:
                     # Clean common prefix for visualization
                     cleaned_token = tokens_raw[i].replace('Ġ', ' ').replace(' ', ' ')
                     tokens_to_viz.append(cleaned_token)
                     activations_to_viz.append(all_activations[i])


        # Log detailed token activations for debugging
        debug_logger = logging.getLogger('debug')
        debug_logger.debug(f"Token activations for feature {feature_idx} in text: {text[:100]}...")
        for token, activation in zip(tokens_to_viz, activations_to_viz):
             if activation > 0: # Only log non-zero activations
                 debug_logger.debug(f"  Token: '{token}', Activation: {activation:.4f}")


        # Skip visualization if no valid tokens remain after filtering
        if not tokens_to_viz:
            debug_logger.warning(f"No non-special tokens to visualize for feature {feature_idx} in text: {text[:50]}...")
            # Create a simple placeholder image
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.text(0.5, 0.5, "No non-special tokens to visualize",
                     ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"Feature {feature_idx} Activation (No Valid Tokens)")
            ax.axis('off')

            # Save placeholder image
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
                img_path = os.path.join(output_dir, img_filename)
            else:
                import tempfile
                temp_dir = tempfile.gettempdir()
                img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
                img_path = os.path.join(temp_dir, img_filename)

            plt.savefig(img_path)
            plt.close(fig) # Close the figure
            return img_path

        # Create plot showing activation intensity with color
        fig, ax = plt.subplots(figsize=(max(len(tokens_to_viz)*0.4, 10), 3))

        # Use a heatmap-style visualization
        # Only positive activations get color
        norm_activations = [max(0, a) for a in activations_to_viz]

        max_activation_val = max(norm_activations) if norm_activations else 1.0

        # Create rectangles for each token
        for i, (token, act) in enumerate(zip(tokens_to_viz, norm_activations)):
            # Normalize activation to [0, 1] range based on the max in this snippet
            color_intensity = min(act / max_activation_val if max_activation_val > 0 else 0, 1.0)
            # Orange color scheme with intensity
            color = (1.0, 0.5 * (1 - color_intensity), 0.2 * (1 - color_intensity)) # Slightly more orange/yellow

            # Use text with background box
            ax.text(i, 0.5, token, ha='center', va='center', fontsize=10,
                    bbox=dict(facecolor=color, alpha=0.7, pad=3, edgecolor='none')) # Use edge color 'none'


        ax.set_xlim(-0.5, len(tokens_to_viz) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_title(f"Feature {feature_idx} Activation")
        ax.axis('off')  # Hide axes

        plt.tight_layout()

        # Save image to output directory or temp dir
        if output_dir:
            # Make sure output_dir exists
            os.makedirs(output_dir, exist_ok=True)
            img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
            img_path = os.path.join(output_dir, img_filename)
        else:
            # Use a temp directory
            import tempfile
            temp_dir = tempfile.gettempdir()
            img_filename = f"feature_{feature_idx}_activation_{int(time.time())}.png"
            img_path = os.path.join(temp_dir, img_filename)

        plt.savefig(img_path)
        plt.close(fig) # Close the figure

        return img_path  # Return path to the saved image

    def compute_feature_statistics(self, feature_idx, texts, n_samples=1000, batch_size=16):
        """Compute statistics for a feature across texts."""
        # Sample texts if needed
        if len(texts) > n_samples:
            sampled_texts = random.sample(texts, n_samples)
        else:
            sampled_texts = texts

        activations = []
        active_token_pct = []

        # Get special tokens to ignore
        special_tokens = set([self.tokenizer.bos_token, self.tokenizer.eos_token,
                              self.tokenizer.pad_token, self.tokenizer.cls_token,
                              self.tokenizer.sep_token, "<s>", "</s>"])
        special_tokens = {t for t in special_tokens if t is not None}
        special_ids = set(self.tokenizer.all_special_ids)

        # Process in batches
        for i in tqdm(range(0, len(sampled_texts), batch_size), desc="Computing statistics", leave=False):
            batch_texts = sampled_texts[i:i+batch_size]
            # Filter empty texts
            batch_texts = [text for text in batch_texts if text.strip()]

            if not batch_texts:
                continue

            # Tokenize batch
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=512).to(self.device)
            token_ids = inputs["input_ids"].cpu().numpy() # Need token_ids for special token check


            with torch.no_grad():
                # Process batch through model
                outputs = self.base_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[self.target_layer]

                # Process through SAE
                features, _ = self.sae_model(hidden_states.to(self.sae_model.encoder[0].weight.dtype))

                # For each sample in batch
                for b in range(features.shape[0]):
                    # Get valid sequence length (exclude padding)
                    seq_len = torch.sum(inputs["attention_mask"][b]).item()

                    if seq_len > 0:  # Skip if sequence is empty
                        # Get activations for this feature
                        token_activations = features[b, :seq_len, feature_idx].cpu().numpy()
                        sample_token_ids = token_ids[b][:seq_len]
                        sample_tokens = self.tokenizer.convert_ids_to_tokens(sample_token_ids)

                        # Create a mask for non-special tokens
                        token_mask = np.ones_like(token_activations, dtype=bool)
                        for k in range(seq_len):
                            if sample_tokens[k] in special_tokens or sample_token_ids[k] in special_ids:
                                token_mask[k] = False

                        # Only consider activations for non-special tokens
                        non_special_activations = token_activations[token_mask]

                        # Accumulate statistics
                        activations.extend(non_special_activations)
                        # Calculate percent active tokens among non-special tokens
                        if np.sum(token_mask) > 0:
                            active_token_pct.append(np.mean(non_special_activations > 0))
                        else:
                            # If only special tokens, 0% active non-special tokens
                            active_token_pct.append(0.0)


        # Convert to numpy arrays
        activations = np.array(activations)
        active_token_pct = np.array(active_token_pct)

        # Compute statistics
        stats = {
            "mean_activation": float(np.mean(activations)) if len(activations) > 0 else 0.0,
            "max_activation": float(np.max(activations)) if len(activations) > 0 else 0.0,
            "median_activation": float(np.median(activations)) if len(activations) > 0 else 0.0,
            "percent_active_non_special_tokens": float(np.mean(activations > 0) * 100) if len(activations) > 0 else 0.0,
            "mean_text_active_pct_non_special_tokens": float(np.mean(active_token_pct) * 100) if len(active_token_pct) > 0 else 0.0,
            "feature_idx": feature_idx
        }

        # Save to instance
        self.feature_stats[feature_idx] = stats

        return stats

    async def identify_feature_pattern(self, feature_idx, example_texts, output_dir=None, example_imgs=None):
            """
            Use Claude API to identify the semantic pattern of a feature.

            Args:
                feature_idx: The feature index
                example_texts: List of example texts (or dicts with text info). Use up to 10 examples.
                output_dir: Directory to save results
                example_imgs: Optional list of image paths (not used by Claude-3 Opus text prompt)

            Returns:
                Identified pattern as a string
            """
            feature_id_logger = logging.getLogger('feature_id')

            # Format examples properly using windowed texts if available
            formatted_examples = []
            for i, example in enumerate(example_texts[:10]):  # Use up to 10 examples
                if isinstance(example, dict) and "windowed_text" in example:
                    text_content = example["windowed_text"]
                else:
                    text_content = example

                text_content = text_content.replace(self.tokenizer.bos_token or "<s>", "") \
                                        .replace(self.tokenizer.eos_token or "</s>", "") \
                                        .replace(self.tokenizer.pad_token or "", "") \
                                        .replace(self.tokenizer.cls_token or "", "") \
                                        .replace(self.tokenizer.sep_token or "", "")
                text_content = text_content.strip()

                if text_content:
                    formatted_examples.append(f"Example {i+1}: {text_content}")

            if not formatted_examples:
                feature_id_logger.warning(f"No valid examples to send to Claude for feature {feature_idx} identification.")
                # Still save prompt and placeholder even if no examples were formatted
                if output_dir:
                    claude_output_dir = os.path.join(output_dir, f"feature_{feature_idx}_examples")
                    os.makedirs(claude_output_dir, exist_ok=True)
                    prompt_path = os.path.join(claude_output_dir, "claude_pattern_prompt.txt")
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write("No valid examples to format for prompt.")
                    response_path = os.path.join(claude_output_dir, "claude_pattern_response.txt")
                    with open(response_path, "w", encoding="utf-8") as f:
                        f.write("No valid examples.")
                    pattern_path = os.path.join(claude_output_dir, "pattern.txt")
                    with open(pattern_path, "w", encoding="utf-8") as f:
                        f.write("Unknown pattern (No valid examples)")

                return "Unknown pattern (No valid examples)"

            # Define the separator just like in your original script
            separator = '-' * 40

            # Create a customized Claude prompt that returns a concise pattern
            # Using the loaded identification_prompt template
            # Pass BOTH 'examples' and 'separator' to the format call
            try:
                claude_prompt = self.identification_prompt.format(
                    examples=chr(10).join(formatted_examples), # Join examples with newline
                    separator=separator # <--- ADDED THIS
                )
            except KeyError as e:
                feature_id_logger.error(f"Prompt template '{self.identification_prompt}' is missing expected placeholder: {e}. Cannot format prompt.")
                if output_dir: # Save error state
                    claude_output_dir = os.path.join(output_dir, f"feature_{feature_idx}_examples")
                    os.makedirs(claude_output_dir, exist_ok=True)
                    error_path = os.path.join(claude_output_dir, "prompt_format_error.txt")
                    with open(error_path, "w", encoding="utf-8") as f:
                        f.write(f"Error formatting prompt: {e}\n")
                        f.write(f"Template:\n---\n{self.identification_prompt}\n---\n")
                        f.write(f"Provided keys: examples, separator\n")
                return f"Error: Prompt template missing placeholder {e}"


            feature_id_logger.debug(f"CLAUDE PATTERN IDENTIFICATION PROMPT (Feature {feature_idx}):\n{claude_prompt}")

            # Ensure output directory exists for Claude response
            if output_dir:
                claude_output_dir = os.path.join(output_dir, f"feature_{feature_idx}_examples")
                os.makedirs(claude_output_dir, exist_ok=True)
            else:
                claude_output_dir = "."


            if not self.claude_api_key:
                feature_id_logger.error("Claude API key is missing. Cannot identify pattern using Claude API.")
                # Save prompt and a placeholder response if key is missing
                prompt_path = os.path.join(claude_output_dir, "claude_pattern_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(claude_prompt) # Save the prompt that *would* have been sent
                response_path = os.path.join(claude_output_dir, "claude_pattern_response.txt")
                with open(response_path, "w", encoding="utf-8") as f:
                    f.write("Error: Claude API key missing.")
                pattern_path = os.path.join(claude_output_dir, "pattern.txt")
                with open(pattern_path, "w", encoding="utf-8") as f:
                    f.write("Unknown pattern (Claude API key missing)")
                return "Unknown pattern (Claude API key missing)"

            # Define Claude API URL and headers
            api_url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.claude_api_key,
                "anthropic-version": "2023-06-01", # Use a recent version
                "content-type": "application/json"
            }

            # Prepare the request data
            data = {
                "model": "claude-3-opus-20240229", # Using Opus as in original script's estimate
                "max_tokens": 500, # Sufficient tokens for pattern + explanation
                "messages": [
                    {"role": "user", "content": claude_prompt}
                ]
            }

            try:
                # Save the prompt before making the call
                prompt_path = os.path.join(claude_output_dir, "claude_pattern_prompt.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(claude_prompt)

                response = requests.post(api_url, headers=headers, json=data)
                response.raise_for_status() # Raise exception for HTTP errors

                response_data = response.json()
                claude_response = response_data["content"][0]["text"]

                feature_id_logger.debug(f"CLAUDE PATTERN IDENTIFICATION RESPONSE (Feature {feature_idx}):\n{claude_response}")

                # Save the raw response
                claude_response_path = os.path.join(claude_output_dir, "claude_pattern_response.txt")
                with open(claude_response_path, "w", encoding="utf-8") as f:
                    f.write(claude_response)

                # Parse the response to extract the pattern
                # Expecting format like "Pattern <concise_pattern> EXPLANATION: <explanation>"
                pattern_parts = claude_response.split("EXPLANATION:", 1)
                concise_pattern = pattern_parts[0].strip() if pattern_parts else "Unknown pattern (parsing failed)"
                explanation = pattern_parts[1].strip() if len(pattern_parts) > 1 else ""

                # Save the parsed pattern
                pattern_path = os.path.join(claude_output_dir, "pattern.txt")
                with open(pattern_path, "w", encoding="utf-8") as f:
                    f.write(f"{concise_pattern}\n{explanation}")

                return concise_pattern # Return the concise pattern


            except Exception as e:
                feature_id_logger.error(f"Error calling Claude API for pattern identification: {e}")
                # Save error response
                if claude_output_dir:
                    error_path = os.path.join(claude_output_dir, "claude_pattern_error.txt")
                    with open(error_path, "w", encoding="utf-8") as f:
                        f.write(f"Error: {e}\n")
                        if 'claude_prompt' in locals(): # Check if prompt was successfully created before error
                            f.write(f"Prompt attempted:\n---\n{claude_prompt}\n---\n")
                return f"Error identifying pattern: {str(e)}"



    async def score_text_relevance(self, pattern, text_snippet, feature_idx, activation_img=None):
        """Use Claude API to score text relevance to pattern on 0-3 scale."""
        # text_snippet is the potentially windowed text extracted based on local activation

        scoring_logger = logging.getLogger('scoring')

        # Ensure pattern is valid before scoring
        if not pattern or "Unknown pattern" in pattern or "Error identifying pattern" in pattern:
             scoring_logger.warning(f"Skipping scoring for feature {feature_idx} due to invalid pattern: '{pattern}'")
             return 1 # Default score for invalid pattern


        if not self.claude_api_key:
            scoring_logger.error("Claude API key is missing. Cannot score relevance using Claude API for feature {feature_idx}.")
            return 1 # Default score

        # Use the existing scoring prompt template
        prompt = self.scoring_prompt.format(
            pattern=pattern,
            text=text_snippet # Use the text snippet extracted previously
        )

        # Add specific instruction for Claude to output JUST the score for reliable parsing
        # This instruction is added *on top* of the template read from scoring_prompt.txt
        prompt = f"{prompt}\n\nRespond ONLY with a single digit (0, 1, 2, or 3).\nScore:" # Added "Score:" hint


        scoring_logger.debug(f"CLAUDE SCORING PROMPT (Feature: {feature_idx}, Pattern: {pattern[:50]}...):\n{prompt}")

        # Define Claude API URL and headers (same as identification)
        api_url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.claude_api_key,
            "anthropic-version": "2023-06-01", # Use a recent version
            "content-type": "application/json"
        }

        # Prepare the request data
        data = {
            "model": "claude-3-haiku-20240307", # Haiku is cheaper and faster for simple scoring
            "max_tokens": 10, # Only need a few tokens for a digit response
            "temperature": 0, # Use low temperature for deterministic scoring
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status() # Raise exception for HTTP errors

            response_data = response.json()
            claude_response_text = response_data["content"][0]["text"]

            scoring_logger.debug(f"CLAUDE SCORING RESPONSE (Feature: {feature_idx}, Pattern: {pattern[:50]}...):\n{claude_response_text}")

            # Parse the score from the response
            # Clean response and look for the digit
            claude_response_text = claude_response_text.strip()
            score_match = re.search(r'[0-3]', claude_response_text)

            if score_match:
                score = int(score_match.group(0))
                scoring_logger.debug(f"Successfully extracted score: {score}")
                return score
            else:
                scoring_logger.warning(f"No score digit found in Claude response: '{claude_response_text}' for prompt starting '{prompt[:100]}...'")
                return 1 # Default to middle-low score if parsing fails

        except Exception as e:
            scoring_logger.error(f"Error calling Claude API for scoring (Feature {feature_idx}): {e}")
            return 1 # Default score on API error


    def clamp_feature_intervention(self, feature_idx, text, clamp_values=None):
        """
        Analyze effect of clamping feature to different values.
        Uses the local base model for generation.

        Args:
            feature_idx: The feature to clamp
            text: Input text prompt
            clamp_values: List of values to clamp the feature to (default: [0.0, 2*max, 5*max])

        Returns:
            Dictionary with generated text for each clamp value
        """
        results = {}

        # Define clamp values if not provided
        if clamp_values is None:
            # Default: zero, 2x max, 5x max activation observed for this feature
            # Fetch max_activation from stored stats if available, otherwise use a fallback
            max_val = self.feature_stats.get(feature_idx, {}).get('max_activation', 1.0)
            clamp_values = [0.0, max_val * 2, max_val * 5]
            print(f"Clamping feature {feature_idx} to default values based on max activation ({max_val:.4f}): {clamp_values}")
        else:
             print(f"Clamping feature {feature_idx} to specified values: {clamp_values}")

        # Get base output (without clamping)
        print("Generating base output (unclamped)...")
        try:
            base_output = self._generate_with_feature_clamped(text, None)
            results["base"] = base_output
        except Exception as e:
            logging.getLogger('main').error(f"Error generating base output for clamping feature {feature_idx}: {e}")
            results["base"] = f"Error generating base output: {str(e)}"


        # Get outputs with clamped values
        for value in clamp_values:
            clamp_dict = {feature_idx: value}
            clamp_key = f"clamp_{value:.4f}" # Use more precise key for float values
            print(f"Generating output with feature {feature_idx} clamped to {value:.4f}...")
            try:
                output = self._generate_with_feature_clamped(text, clamp_dict)
                results[clamp_key] = output
            except Exception as e:
                 logging.getLogger('main').error(f"Error generating clamped output ({value:.4f}) for feature {feature_idx}: {e}")
                 results[clamp_key] = f"Error generating output clamped to {value:.4f}: {str(e)}"


        return results

# Replace the previous _generate_with_feature_clamped method with this corrected version

    def _generate_with_feature_clamped(self, text, clamp_features=None):
        """
        Generate text with features clamped to specific values using the local base model.

        Args:
            text: Input text
            clamp_features: Dict mapping {feature_idx: value} or None

        Returns:
            Generated text
        """
        # Tokenize the input text
        # Use padding=True for consistent input shape if needed, though not strictly required for batch_size 1 here
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        # Set up hooks for clamping features during generation
        hooks = []
        hook = None # Initialize hook outside the if block

        if clamp_features:
            # Define the forward hook that modifies SAE features
            # This hook needs to be registered with the SAE model's encoder output

            def forward_hook(module, input_tensor, output):
                # The SAE forward is expected to return (features, reconstruction)
                features, reconstruction = output

                # Clamp specified features IN PLACE on the 'features' tensor
                # The shape of features is (batch_size, seq_len, hidden_dim)
                # Assuming batch_size is 1 during generation for a single prompt
                for feat_idx, value in clamp_features.items():
                    if feat_idx < features.shape[-1]: # Ensure feature index is valid within the hidden dimension
                         # Clamp the feature activation value across all tokens in the sequence for this batch item (index 0)
                         features[:, :, feat_idx] = value # Modify features in place
                    else:
                         logging.getLogger('debug').warning(f"Attempted to clamp invalid feature index {feat_idx}. Max feature index is {features.shape[-1]-1}")


                # Return the modified features and the original reconstruction
                # The base model generation process will continue using these modified features
                return features, reconstruction

            # Find the SAE encoder layer to hook into.
            # Based on the class definition, the encoder is a Sequential, and the activation
            # happens after the linear layer. Hooking the Sequential itself or the ReLU layer
            # should give access to the post-activation features. Let's hook the Sequential encoder.
            try:
                 # Register the hook on the SAE's encoder Sequential module
                 hook = self.sae_model.encoder.register_forward_hook(forward_hook)
                 hooks.append(hook)
                 logging.getLogger('debug').debug(f"Registered forward hook on SAE encoder for clamping.")
            except Exception as e:
                 logging.getLogger('main').error(f"Failed to register forward hook on SAE encoder for clamping: {e}")
                 # If hook registration fails, continue without clamping, but log it


        # Generate with hooks active
        result = ""
        try:
            with torch.no_grad():
                # When using generate with hooks that modify intermediate states,
                # HuggingFace's generate needs output_hidden_states=True to ensure the model
                # passes through the layers where hooks are registered (which happens during the forward pass).
                outputs = self.base_model.generate(
                    **inputs, # Pass input_ids and attention_mask from tokenization
                    max_new_tokens=300, # Increase max tokens for generated response length
                    do_sample=True,
                    temperature=0.7,
                    return_dict_in_generate=True, # Returns a dictionary-like object
                    output_hidden_states=True, # Essential to trigger hooks modifying hidden states during forward pass
                    # DO NOT pass attention_mask separately here, it's already in **inputs
                    # attention_mask=inputs.get('attention_mask', None) # <-- REMOVE THIS LINE
                )

            # Get the generated tokens from the output object
            # Access the sequences attribute when return_dict_in_generate=True
            generated_sequence = outputs.sequences[0]

            # Decode the generated sequence, skipping special tokens
            # Start decoding AFTER the input prompt tokens to get only the new text
            input_len = inputs['input_ids'].shape[1]
            # Ensure generated_sequence is longer than input_len before slicing to avoid index errors
            if generated_sequence.shape[0] > input_len:
                result = self.tokenizer.decode(generated_sequence[input_len:], skip_special_tokens=True)
            else:
                # This case happens if the model generates 0 new tokens
                result = "" # No new tokens were generated

        except Exception as e:
            logging.getLogger('main').error(f"Error during text generation with clamping hook: {e}")
            result = f"Error during generation: {str(e)}"
        finally:
            # Always remove hooks to avoid side effects on subsequent generations
            # Only remove the hook if it was successfully registered
            if hook is not None:
                 hook.remove()
                 logging.getLogger('debug').debug(f"Removed generation hook.")


        return result
    
async def main(): # Keep main async because we now have async calls inside

    parser = argparse.ArgumentParser(description="Run Anthropic-style SAE feature analysis")
    parser.add_argument("--model_path", type=str, required=True, help="Path to base LLM model")
    parser.add_argument("--sae_path", type=str, required=True, help="Path to trained SAE model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of text samples to use from dataset")
    parser.add_argument("--num_features", type=int, default=10, help="Number of features to analyze")
    parser.add_argument("--num_examples", type=int, default=20, help="Number of top examples to find per feature")
    parser.add_argument("--num_scoring_samples", type=int, default=100,
                        help="Number of text samples to score for specificity analysis (if --run_scoring is used)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for processing texts")
    parser.add_argument("--num_initial_samples", type=int, default=1000,
                        help="Number of samples to use for initial feature discovery")
    parser.add_argument("--activation_threshold", type=float, default=10.0,
                        help="Minimum activation threshold for feature selection")
    parser.add_argument("--max_activation_threshold", type=float, default=None,
                        help="Maximum activation threshold for feature selection (no upper limit if not set)")
    parser.add_argument("--window_size", type=int, default=10,
                        help="Number of tokens before and after highest activation to include (max 10)")
    parser.add_argument("--config_dir", type=str, default="../config",
                        help="Directory containing API keys configuration")
    # Add the argument to control scoring run - defaulted to skip
    parser.add_argument(
        "--run_scoring",
        action='store_true', # This flag will be True if provided, False otherwise
        default=False,       # Default behavior is to NOT run scoring (i.e., skip scoring)
        help="Set this flag to run the feature scoring and specificity analysis using Claude API (off by default)."
    )

    args = parser.parse_args()

    # Limit window size to maximum of 10
    args.window_size = min(args.window_size, 10)

    # Load Claude API key from config directory
    claude_api_key = None
    config_path = Path(args.config_dir) / "api_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                claude_api_key = config.get("claude_api_key")
        except Exception as e:
            print(f"Warning: Failed to load Claude API key from config '{config_path}': {e}")

    if not claude_api_key:
         print("Warning: Claude API key not found. Feature identification and scoring (if enabled) will likely fail.")


    # Estimate API calls and give the user a chance to cancel
    estimated_id_calls = args.num_features
    estimated_scoring_calls = args.num_features * args.num_scoring_samples if args.run_scoring else 0
    estimated_total_calls = estimated_id_calls + estimated_scoring_calls

    print(f"\n===== CLAUDE API USAGE ESTIMATE =====")
    print(f"Feature Identification: {estimated_id_calls} calls ({args.num_features} features)")
    print(f"Specificity Scoring (--run_scoring {'enabled' if args.run_scoring else 'disabled'}): {estimated_scoring_calls} calls ({args.num_features} features * {args.num_scoring_samples} samples/feature)")
    print(f"Total Estimated Calls: {estimated_total_calls}")

    # Use the same $0.08 estimate as in the original script for Opus identification calls.
    # Note: Scoring calls now use Haiku (cheaper), but using the Opus rate gives a conservative upper bound for the estimate.
    estimated_cost = estimated_total_calls * 0.08

    print(f"Estimated API cost (assuming $0.08/call for simplicity): ${estimated_cost:.2f}")
    print(f"=====================================\n")

    # Ask user for confirmation ONLY if any API calls are expected AND we have a key
    if estimated_total_calls > 0 and claude_api_key:
         response = input("Continue with this API usage? (y/n): ")
         if response.lower() != 'y':
             print("Run canceled by user.")
             return
    elif estimated_total_calls > 0 and not claude_api_key:
         print("WARNING: Claude API calls estimated, but no API key found. API steps will likely fail.")
         # Optionally ask to continue anyway if they understand API calls won't work
         response = input("Continue anyway? (API calls will fail) (y/n): ")
         if response.lower() != 'y':
              print("Run canceled by user.")
              return


    # Load dataset
    print(f"Loading wikitext dataset with {args.num_samples} samples...")
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Take a subset of texts for analysis
        texts = [sample["text"] for sample in dataset.select(range(min(len(dataset), args.num_samples)))]
        texts = [text for text in texts if len(text.strip()) > 0]  # Filter empty texts
        print(f"Loaded {len(texts)} text samples from wikitext")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure you have an internet connection to download the dataset or use a local path.")
        return # Exit if dataset loading fails


    # Create output directory with timestamp and prompt directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    # Check if output_dir exists and is writable
    if not base_output_dir.exists():
        try:
            base_output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             print(f"Error creating output directory '{base_output_dir}': {e}")
             print("Please check directory path and permissions.")
             return

    # Create timestamped subdirectory
    output_dir = base_output_dir / f"run_{timestamp}"
    try:
        output_dir.mkdir(exist_ok=True)
    except OSError as e:
         print(f"Error creating run directory '{output_dir}': {e}")
         print("Please check directory path and permissions.")
         return


    # Create prompts directory if it doesn't exist and check prompt files
    prompts_dir = Path("prompts")
    prompts_dir.mkdir(exist_ok=True) # Create prompts dir if it doesn't exist

    identification_prompt_path = prompts_dir / "feature_identification_prompt.txt"
    scoring_prompt_path = prompts_dir / "scoring_prompt.txt"

    IDENTIFICATION_PROMPT = ""
    if identification_prompt_path.exists():
        with open(identification_prompt_path, "r", encoding="utf-8") as f:
             IDENTIFICATION_PROMPT = f.read()
             if "{examples}" not in IDENTIFICATION_PROMPT:
                  print(f"Warning: Prompt template '{identification_prompt_path}' does not contain '{{examples}}'. Pattern identification may fail.")
             print(f"Loaded prompt template: {identification_prompt_path}")
    else:
        print(f"Error: Prompt template '{identification_prompt_path}' not found.")
        print("Please create this file with the prompt for Claude feature identification.")
        return # Exit if prompts are missing

    SCORING_PROMPT = ""
    if scoring_prompt_path.exists():
         with open(scoring_prompt_path, "r", encoding="utf-8") as f:
              SCORING_PROMPT = f.read()
              if "{pattern}" not in SCORING_PROMPT or "{text}" not in SCORING_PROMPT:
                   print(f"Warning: Prompt template '{scoring_prompt_path}' does not contain '{{pattern}}' or '{{text}}'. Scoring may fail.")
              print(f"Loaded prompt template: {scoring_prompt_path}")
    else:
        print(f"Error: Prompt template '{scoring_prompt_path}' not found.")
        print("Please create this file with the prompt for Claude scoring.")
        return # Exit if prompts are missing


    # Set up logging with separate files for different components
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Main log file
    main_log_file = logs_dir / "main.log"
    main_handler = logging.FileHandler(str(main_log_file), encoding='utf-8')
    main_handler.setLevel(logging.INFO)
    main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    main_handler.setFormatter(main_formatter)

    # Feature identification log
    feature_id_log_file = logs_dir / "feature_identification.log"
    feature_id_handler = logging.FileHandler(str(feature_id_log_file), encoding='utf-8')
    feature_id_handler.setLevel(logging.DEBUG)
    feature_id_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    feature_id_handler.setFormatter(feature_id_formatter)

    # Scoring log
    scoring_log_file = logs_dir / "scoring.log"
    scoring_handler = logging.FileHandler(str(scoring_log_file), encoding='utf-8')
    scoring_handler.setLevel(logging.DEBUG)
    scoring_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    scoring_handler.setFormatter(scoring_formatter)

    # Other debug information
    debug_log_file = logs_dir / "debug.log"
    debug_handler = logging.FileHandler(str(debug_log_file), encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)

    # Console output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s') # Simple format for console
    console.setFormatter(console_formatter)

    # Create loggers
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    main_logger.addHandler(main_handler)
    main_logger.addHandler(console) # Add console handler

    feature_id_logger = logging.getLogger('feature_id')
    feature_id_logger.setLevel(logging.DEBUG)
    feature_id_logger.addHandler(feature_id_handler)

    scoring_logger = logging.getLogger('scoring')
    scoring_logger.setLevel(logging.DEBUG)
    scoring_logger.addHandler(scoring_handler)

    debug_logger = logging.getLogger('debug')
    debug_logger.setLevel(logging.DEBUG)
    debug_logger.addHandler(debug_handler)


    main_logger.info(f"Results will be saved to: {output_dir}")
    debug_logger.debug("Logging initialized with separate log files")


    # Save run parameters for reproducibility
    try:
        with open(output_dir / "run_parameters.json", "w") as f:
            params = {
                "model_path": args.model_path,
                "sae_path": args.sae_path,
                "num_samples": args.num_samples,
                "num_features": args.num_features,
                "num_examples": args.num_examples,
                "num_scoring_samples": args.num_scoring_samples,
                "batch_size": args.batch_size,
                "num_initial_samples": args.num_initial_samples,
                "activation_threshold": args.activation_threshold,
                "max_activation_threshold": args.max_activation_threshold,
                "window_size": args.window_size,
                "using_claude_api_identification": claude_api_key is not None,
                "running_scoring": args.run_scoring, # Indicate if scoring was run
                "timestamp": timestamp
            }
            json.dump(params, f, indent=2)
    except Exception as e:
        main_logger.warning(f"Failed to save run parameters: {e}")


    # Load models
    print("Loading models...")
    try:
        # Use the same tokenizer configuration as in the SAE training script
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=False,  # Use Python implementation instead of Rust
            padding_side="right",
            local_files_only=True # Ensure loading from local files only
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Often needed for generation/padding
    except Exception as e:
        print(f"Error loading tokenizer from '{args.model_path}': {e}")
        return

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            torch_dtype=torch.float16, # Use half-precision to save memory
            local_files_only=True # Ensure loading from local files only
        )
        base_model.eval()
    except Exception as e:
        print(f"Error loading base model from '{args.model_path}': {e}")
        return

    # Load SAE model
    print("Loading SAE model...")
    try:
        # Load state dict first to infer dimensions
        state_dict = torch.load(args.sae_path, map_location=base_model.device)

        # Determine dimensions from the state dict
        input_dim = None
        hidden_dim = None

        if 'decoder.weight' in state_dict:
            decoder_weight = state_dict['decoder.weight']
            input_dim = decoder_weight.shape[0]
            hidden_dim = decoder_weight.shape[1]
        elif 'encoder.0.weight' in state_dict:
             encoder_weight = state_dict['encoder.0.weight']
             hidden_dim, input_dim = encoder_weight.shape
        else:
             raise ValueError("Could not determine SAE dimensions from state dict keys (expected 'decoder.weight' or 'encoder.0.weight')")

        print(f"Creating SAE with input_dim={input_dim}, hidden_dim={hidden_dim}")
        sae_model = SparseAutoencoder(input_dim, hidden_dim)

        # Load the state dict into the created model instance
        missing_keys, unexpected_keys = sae_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys in SAE state dict: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys in SAE state dict: {unexpected_keys}")

        sae_model.to(base_model.device)
        sae_model.eval()

    except FileNotFoundError:
        print(f"Error: SAE file not found at '{args.sae_path}'.")
        return
    except ValueError as ve:
         print(f"Error loading SAE model: {ve}")
         return
    except Exception as e:
        print(f"Error loading SAE model from '{args.sae_path}': {e}")
        return


    # Initialize analyzer
    analyzer = SAEFeatureAnalyzer(
        sae_model,
        tokenizer,
        base_model, # base_model instance is needed by the analyzer
        identification_prompt=IDENTIFICATION_PROMPT,
        scoring_prompt=SCORING_PROMPT,
        claude_api_key=claude_api_key
    )

    # Find active features above threshold
    print("Finding features within activation range...")
    start_time = time.time()
    active_features = analyzer.find_most_active_features(
        texts,
        n_features=args.num_features,
        n_samples=args.num_initial_samples,
        batch_size=args.batch_size,
        activation_threshold=args.activation_threshold,
        max_activation_threshold=args.max_activation_threshold
    )
    feature_discovery_time = time.time() - start_time

    print(f"Feature discovery took {feature_discovery_time:.2f} seconds")
    print(f"Selected {len(active_features)} features by activation:")
    if active_features:
        for feature_idx, max_activation in active_features:
            print(f"Feature {feature_idx}: Max activation = {max_activation:.4f}")
    else:
        print("No features found meeting the specified criteria. Exiting analysis.")
        main_logger.info("No features found meeting criteria. Analysis stopped.")
        return # Exit if no features are found

    # Get the main logger for general progress information
    main_logger = logging.getLogger('main')

    # Analyze each feature
    results = []

    for i, (feature_idx, _) in enumerate(active_features):
        print(f"\n{'-'*80}\nAnalyzing feature {feature_idx} ({i+1}/{len(active_features)})")
        main_logger.info(f"Analyzing feature {feature_idx} ({i+1}/{len(active_features)})")
        feature_result = {"feature_idx": int(feature_idx)} # Ensure feature_idx is serializable

        # Find top examples with window around max activation
        print(f"Finding top {args.num_examples} examples...")
        main_logger.info(f"Finding top {args.num_examples} examples for feature {feature_idx}...")
        start_time = time.time()
        top_examples = analyzer.find_highest_activating_examples(
            feature_idx,
            texts,
            top_n=args.num_examples,
            batch_size=args.batch_size,
            window_size=args.window_size # Pass window_size
        )
        example_time = time.time() - start_time
        print(f"Example collection took {example_time:.2f} seconds")

        # Store text examples and create individual visualization images
        example_texts = []
        examples_dir = output_dir / f"feature_{feature_idx}_examples"
        examples_dir.mkdir(exist_ok=True)
        example_imgs = [] # List to store paths to visualization images

        print("Creating token-level visualizations for examples...")
        # Save all examples to a single text file for easy viewing
        try:
            with open(examples_dir / "all_examples_raw.txt", "w", encoding="utf-8") as f:
                for j, example in enumerate(top_examples):
                    f.write(f"--- Example {j+1} ---\n")
                    f.write(f"Max Activation: {example['max_activation']:.4f}\n")
                    f.write(f"Max Token Position: {example['max_position']} (Window: {example['window_start']}-{example['window_end']})\n")
                    f.write(f"Max Token: '{example['max_token']}'\n")
                    f.write(f"Windowed Text:\n{example['windowed_text']}\n")
                    f.write(f"Full Text:\n{example['text']}\n\n")
        except Exception as e:
             main_logger.warning(f"Failed to save all_examples_raw.txt for feature {feature_idx}: {e}")


        for j, example in enumerate(top_examples):
            # Pass output directory and window_size to visualization function
            img_path = analyzer.visualize_token_activations(
                example["windowed_text"], # Visualize the windowed text snippet
                feature_idx,
                output_dir=examples_dir,
                window_size=args.window_size # This arg is used internally by visualize_token_activations if text is not windowed
            )
            # Store the relative path name
            example_imgs.append(os.path.basename(img_path))

            # Also save the raw text for reference alongside the image
            try:
                with open(examples_dir / f"example_{j+1}.txt", "w", encoding="utf-8") as f:
                    f.write(f"Full text:\n{example['text']}\n\n")
                    f.write(f"Windowed text ({args.window_size} tokens before and after max activation):\n{example['windowed_text']}\n\n")
                    f.write(f"Max activation: {example['max_activation']:.4f}\n")
                    f.write(f"Max token: '{example['max_token']}'\n")
                    f.write(f"Token position (within original text): {example['max_position']} (window: {example['window_start']}-{example['window_end']})")
            except Exception as e:
                 main_logger.warning(f"Failed to save example text file {j+1} for feature {feature_idx}: {e}")


        # Store cleaned example texts (just the windowed snippets) for pattern identification
        feature_result["examples"] = [{"windowed_text": ex["windowed_text"],
                                      "max_activation": float(ex["max_activation"]), # Ensure serializable
                                      "max_token": ex["max_token"]} for ex in top_examples]
        feature_result["example_imgs"] = example_imgs


        # Compute feature statistics
        print("Computing feature statistics...")
        main_logger.info(f"Computing statistics for feature {feature_idx}...")
        stats = analyzer.compute_feature_statistics(
            feature_idx,
            texts,
            n_samples=min(2000, len(texts)), # Use up to 2000 samples for stats
            batch_size=args.batch_size
        )
        feature_result["statistics"] = stats
        print(f"Statistics: {stats}")


        # Identify pattern using Claude API
        print("Identifying feature pattern using Claude API...")
        main_logger.info(f"Identifying pattern for feature {feature_idx} using Claude API...")
        # Pass the windowed examples to the pattern identification
        # Using async and await now
        feature_result["pattern"] = await analyzer.identify_feature_pattern(
             feature_idx,
             feature_result["examples"], # Use the extracted examples
             output_dir=output_dir
             # example_imgs are not used by the current text-only prompt for Claude Opus
         )
        print(f"Identified pattern: {feature_result['pattern']}")


        # --- Specificity Scoring Block (NOW CONDITIONAL AND USES CLAUDE) ---
        if args.run_scoring:
            print(f"Scoring {args.num_scoring_samples} texts for specificity using Claude API...")
            main_logger.info(f"Scoring {args.num_scoring_samples} texts for specificity with pattern '{feature_result['pattern']}' using Claude API...")

            if not claude_api_key:
                 print("Skipping specificity scoring because Claude API key is missing.")
                 main_logger.warning("Skipping specificity scoring because Claude API key is missing.")
                 feature_result["specificity_scores"] = []
                 feature_result["specificity_activations"] = []
                 feature_result["histogram_plot"] = None
                 feature_result["specificity_plot"] = None
                 # No need to compute activations or score if key is missing

            elif "Unknown pattern" in feature_result['pattern'] or "Error identifying pattern" in feature_result['pattern']:
                 print(f"Skipping specificity scoring for feature {feature_idx} due to invalid pattern.")
                 main_logger.warning(f"Skipping specificity scoring for feature {feature_idx} due to invalid pattern: '{feature_result['pattern']}'")
                 feature_result["specificity_scores"] = []
                 feature_result["specificity_activations"] = []
                 feature_result["histogram_plot"] = None
                 feature_result["specificity_plot"] = None
            else:
                 # Select texts for scoring (could be same as or subset of overall texts)
                 if len(texts) > args.num_scoring_samples:
                     scoring_texts_full = random.sample(texts, args.num_scoring_samples)
                 else:
                     scoring_texts_full = texts


                 # Get activations for scoring texts and extract snippets to send to Claude
                 print("Computing activations for scoring texts and extracting snippets...")
                 scoring_data = [] # Will store {"windowed_text": ..., "max_activation": ...}
                 for i in tqdm(range(0, len(scoring_texts_full), args.batch_size), desc="Extracting snippets for scoring"):
                     batch_texts = scoring_texts_full[i:i+args.batch_size]
                     batch_texts = [text for text in batch_texts if text.strip()]

                     if not batch_texts:
                         continue

                     # Tokenize batch and process through model/SAE to get activations
                     inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                                        truncation=True, max_length=512).to(base_model.device)
                     token_ids = inputs["input_ids"].cpu().numpy()

                     with torch.no_grad():
                         outputs = base_model(**inputs, output_hidden_states=True)
                         hidden_states = outputs.hidden_states[analyzer.target_layer]
                         features, _ = sae_model(hidden_states.to(sae_model.encoder[0].weight.dtype))

                         # For each sample in batch, find max activation token and extract windowed text
                         for b, text in enumerate(batch_texts):
                             seq_len = torch.sum(inputs["attention_mask"][b]).item()
                             if seq_len == 0: continue

                             token_activations = features[b, :seq_len, feature_idx]

                             # Find max activation position among non-special tokens in this snippet
                             sample_token_ids = token_ids[b][:seq_len]
                             tokens_list = tokenizer.convert_ids_to_tokens(sample_token_ids)
                             special_tokens = set([tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token, "<s>", "</s>"])
                             special_tokens = {t for t in special_tokens if t is not None}
                             special_ids = set(tokenizer.all_special_ids)

                             token_mask = torch.ones_like(token_activations, dtype=torch.bool)
                             for k in range(seq_len):
                                  if tokens_list[k] in special_tokens or sample_token_ids[k] in special_ids:
                                      token_mask[k] = False

                             masked_activations = token_activations.clone()
                             masked_activations[~token_mask] = -1e9 # Set to very low value to ignore special tokens

                             max_activation_val = torch.max(masked_activations).item()

                             # Only add for scoring if there was meaningful activation on non-special tokens
                             if max_activation_val > 0:
                                  max_position = torch.argmax(masked_activations).item()
                                  window_size = args.window_size
                                  start_pos = max(0, max_position - window_size)
                                  end_pos = min(seq_len, max_position + window_size + 1)
                                  window_token_ids = inputs["input_ids"][b, start_pos:end_pos]
                                  # Decode with skip_special_tokens=True for the text sent to Claude
                                  windowed_text = tokenizer.decode(window_token_ids, skip_special_tokens=True)
                                  windowed_text = windowed_text.strip() # Strip after decoding

                                  if windowed_text: # Only score if snippet is not empty after stripping
                                      scoring_data.append({
                                          "text": text, # Original text (for reference/debugging)
                                          "windowed_text": windowed_text, # Snippet to send to Claude
                                          "max_activation": max_activation_val # Use this max activation for the plot later
                                      })


                 # Now score the extracted snippets with Claude API one by one
                 print(f"Scoring {len(scoring_data)} extracted snippets with Claude API...")
                 scores = []
                 activation_values = [] # Store corresponding activation values

                 if not scoring_data:
                      print("No valid snippets extracted for scoring after filtering.")
                      main_logger.warning(f"Feature {feature_idx}: No valid snippets extracted for scoring.")
                      feature_result["specificity_scores"] = []
                      feature_result["specificity_activations"] = []

                 else:
                     for data in tqdm(scoring_data, desc="Scoring with Claude API"):
                          # Score relevance using the extracted windowed text snippet
                          score = await analyzer.score_text_relevance(
                              feature_result['pattern'],
                              data["windowed_text"], # Use the extracted text snippet
                              feature_idx=feature_idx # Pass feature_idx for logging
                          )
                          scores.append(score)
                          activation_values.append(data["max_activation"]) # Use the max activation value from the original snippet extraction

                     feature_result["specificity_scores"] = scores
                     feature_result["specificity_activations"] = activation_values

                     # Create visualization plots (still use matplotlib based on collected scores/activations)
                     print("Creating specificity visualization plots...")
                     main_logger.info(f"Creating specificity visualization plots for feature {feature_idx}...")

                     # Activation histogram (uses activations from extracted snippets used for scoring)
                     # Use activation_values collected from scoring_data as the data for the histogram
                     hist_fig = analyzer.create_activation_histogram(
                         feature_idx,
                         # Need to pass texts again to compute histogram, or use the collected activation_values
                         # Recomputing seems safer to get a distribution over the full set of scoring_texts_full
                         # Or, plot the distribution of the 'max_activation' values collected in scoring_data
                         # Let's plot the distribution of collected 'max_activation' values for the snippets that were scored
                         # Pass the list of max_activation values directly
                         texts=scoring_texts_full, # Pass the original texts to recompute activations for histogram
                         title=f"Pattern: {feature_result['pattern'][:50]}...",
                         batch_size=args.batch_size # Keep batching for computing
                     )
                     hist_fig_path = output_dir / f"feature_{feature_idx}_histogram.png"
                     try:
                         hist_fig.savefig(hist_fig_path)
                         feature_result["histogram_plot"] = str(hist_fig_path.name) # Store filename
                     except Exception as e:
                         main_logger.warning(f"Failed to save histogram plot for feature {feature_idx}: {e}")
                         feature_result["histogram_plot"] = None
                     plt.close(hist_fig) # Close the figure to free memory


                     # Specificity plot (uses scores from Claude and max activations from local model)
                     if scores and activation_values: # Only plot if we have data points
                         spec_fig = analyzer.plot_specificity_scores(
                             feature_idx,
                             scores,
                             activation_values,
                             title=f"Pattern: {feature_result['pattern'][:50]}..."
                         )
                         spec_fig_path = output_dir / f"feature_{feature_idx}_specificity.png"
                         try:
                             spec_fig.savefig(spec_fig_path)
                             feature_result["specificity_plot"] = str(spec_fig_path.name) # Store filename
                         except Exception as e:
                             main_logger.warning(f"Failed to save specificity plot for feature {feature_idx}: {e}")
                             feature_result["specificity_plot"] = None
                         plt.close(spec_fig) # Close the figure
                     else:
                          main_logger.warning(f"Skipping specificity plot for feature {feature_idx} as no score/activation data points were collected.")
                          feature_result["specificity_plot"] = None


        else: # if not args.run_scoring
            print("Skipping specificity scoring and related plots as --run_scoring flag was not set.")
            main_logger.info(f"Skipping specificity scoring for feature {feature_idx}.")
            feature_result["specificity_scores"] = None # Store None if skipped
            feature_result["specificity_activations"] = None
            feature_result["histogram_plot"] = None
            feature_result["specificity_plot"] = None


        # --- End Specificity Scoring Block ---


        # Clamping intervention experiments (STILL USES LOCAL MODEL)
        print("Running clamping interventions (using local model)...")
        main_logger.info(f"Running clamping interventions for feature {feature_idx} (using local model)...")
        # Use a simple conversational prompt
        intervention_prompt = "Human: What's your favorite animal?\n\nAssistant:"
        # Note: clamp_feature_intervention and _generate_with_feature_clamped still use base_model
        try:
            clamp_results = analyzer.clamp_feature_intervention(feature_idx, intervention_prompt)
            feature_result["clamp_results"] = clamp_results
            print(f"Clamping results for feature {feature_idx} obtained.")
        except Exception as e:
             main_logger.error(f"Error running clamping intervention for feature {feature_idx}: {e}")
             feature_result["clamp_results"] = {"error": str(e)}
             print(f"Error running clamping intervention for feature {feature_idx}: {e}")



        # Store all results for this feature
        results.append(feature_result)

        # Save intermediate results (ensure JSON serialization handles None and NumPy)
        try:
            with open(output_dir / "analysis_results.json", "w", encoding="utf-8") as f:
                 json_results = []
                 for res in results:
                     json_res = {}
                     for k, v in res.items():
                          # Handle None, NumPy types, and potential plots which are just filenames now
                          if v is None or isinstance(v, (str, int, float, bool)):
                              json_res[k] = v
                          elif isinstance(v, dict):
                              clean_dict = {}
                              for dict_k, dict_v in v.items():
                                  if hasattr(dict_v, 'item') and callable(getattr(dict_v, 'item')):
                                      clean_dict[dict_k] = dict_v.item()
                                  elif isinstance(dict_v, (np.integer, np.floating, np.bool_)):
                                       clean_dict[dict_k] = dict_v.item()
                                  else:
                                      clean_dict[dict_k] = dict_v
                              json_res[k] = clean_dict
                          elif isinstance(v, list):
                               json_res[k] = [
                                   x.item() if hasattr(x, 'item') and callable(getattr(x, 'item')) else x
                                   for x in v
                               ]
                          else:
                              # Should not happen with the current types, but as a fallback
                              main_logger.warning(f"Unexpected type for JSON serialization: {type(v)} for key {k}")
                              json_res[k] = str(v) # Attempt to convert to string

                     json_results.append(json_res)

                 json.dump(json_results, f, indent=2, ensure_ascii=False)

            print(f"Intermediate results saved for feature {feature_idx}.")
            main_logger.info(f"Intermediate results saved for feature {feature_idx}.")

        except Exception as e:
             main_logger.error(f"Failed to save intermediate JSON results for feature {feature_idx}: {e}")
             print(f"Error saving intermediate results: {e}")


    # Create summary visualizations (e.g., overall stats, if needed)
    print("\nCreating analysis summary...")
    main_logger.info("Creating analysis summary.")

    # Save overall summary (Markdown file)
    try:
        with open(output_dir / "analysis_summary.md", "w", encoding="utf-8") as f:
            f.write("# SAE Feature Analysis Summary\n\n")
            f.write(f"Run Timestamp: {timestamp}\n")
            f.write(f"Base Model: {args.model_path}\n")
            f.write(f"SAE Model: {args.sae_path}\n")
            f.write(f"Parameters: {json.dumps(params, indent=2)}\n\n")
            f.write("---\n\n")

            if not active_features:
                 f.write("No features found meeting the specified criteria.\n\n")
            else:
                f.write(f"## Analyzed Features ({len(active_features)})\n\n")
                for feature_result in results:
                    feature_idx = feature_result.get("feature_idx", "N/A")
                    pattern = feature_result.get("pattern", "N/A Pattern")
                    stats = feature_result.get("statistics", {})
                    clamp_results = feature_result.get("clamp_results", {})

                    f.write(f"### Feature {feature_idx}\n\n")
                    f.write(f"**Pattern:** {pattern}\n\n")

                    f.write(f"**Statistics:**\n")
                    f.write(f"- Max activation: {stats.get('max_activation', 0.0):.4f}\n")
                    f.write(f"- Mean activation: {stats.get('mean_activation', 0.0):.4f}\n")
                    f.write(f"- Percent active tokens (non-special): {stats.get('percent_active_non_special_tokens', 0.0):.2f}%\n\n") # Updated stat name

                    # Link to example details and images
                    f.write(f"**Top Examples:** ([Details & Visualizations](feature_{feature_idx}_examples/))\n\n")
                    # Embed first few examples and images if available
                    examples_list = feature_result.get("examples", [])
                    example_imgs_list = feature_result.get("example_imgs", [])

                    for i, example in enumerate(examples_list[:3]): # Show top 3 in summary
                         f.write(f"Example {i+1}:\n")
                         # Markdown code block for windowed text
                         f.write(f"```\n{example.get('windowed_text', 'N/A')}\n```\n\n")
                         f.write(f"Max token: '{example.get('max_token', 'N/A')}', Activation: {example.get('max_activation', 0.0):.4f}\n\n")
                         # Embed corresponding image if exists
                         if i < len(example_imgs_list) and example_imgs_list[i]:
                              f.write(f"![Activation Visualization](feature_{feature_idx}_examples/{example_imgs_list[i]})\n\n")

                    # Specificity Analysis Summary
                    f.write(f"**Specificity Analysis:**\n")
                    if args.run_scoring:
                         scores = feature_result.get("specificity_scores")
                         activations = feature_result.get("specificity_activations")
                         hist_plot_file = feature_result.get("histogram_plot")
                         spec_plot_file = feature_result.get("specificity_plot")

                         if scores is not None and activations is not None:
                             f.write(f"- Scored {len(scores)} samples.\n")
                             f.write(f"- Mean Score: {np.mean(scores):.2f}" if scores else "- Mean Score: N/A")
                             f.write(f", Percentage with Score 3: {(np.mean(np.array(scores) == 3) * 100):.2f}%" if scores else "")
                             f.write("\n\n")
                             if hist_plot_file:
                                  f.write(f"![Activation Histogram]({hist_plot_file})\n\n")
                             if spec_plot_file:
                                  f.write(f"![Specificity Scores vs Activation]({spec_plot_file})\n\n")
                         else:
                             f.write("- Specificity analysis skipped or failed.\n\n")
                    else:
                         f.write("- Specificity analysis skipped (--run_scoring not set).\n\n")


                    # Clamping Intervention Summary
                    f.write("**Clamping Intervention:**\n")
                    if clamp_results:
                         f.write("Text generated with features clamped to different values.\n")
                         # Show Base vs Clamped 0.0
                         base_text = clamp_results.get("base", "N/A")
                         clamp_zero_text = clamp_results.get("clamp_0.0", "N/A")
                         f.write(f"- **Base (Unclamped):**\n```\n{base_text}\n```\n\n")
                         f.write(f"- **Clamped to 0.0:**\n```\n{clamp_zero_text}\n```\n\n")
                         # Add note about other clamp values if present
                         other_clamps = {k: v for k, v in clamp_results.items() if k not in ["base", "clamp_0.0", "error"]}
                         if other_clamps:
                             f.write(f"- *Other clamp values tested (e.g., {list(other_clamps.keys())[0].split('_')[1]}x max activation). See results JSON for full output.*\n\n")
                         elif "error" in clamp_results:
                              f.write(f"- Error during clamping: {clamp_results['error']}\n\n")

                    else:
                         f.write("- Clamping intervention skipped or failed.\n\n")

                    f.write("---\n\n")

    except Exception as e:
        main_logger.error(f"Failed to generate analysis summary Markdown: {e}")
        print(f"Error generating summary: {e}")


    print("\nAnalysis complete!")
    main_logger.info(f"Results saved to: {output_dir}")
    main_logger.info("Analysis finished successfully.")
    print("Please check the output directory for results, logs, and images.")


if __name__ == "__main__":
    # Ensure matplotlib is not using a GUI backend in non-interactive mode
    plt.switch_backend('Agg') # Use Agg backend (non-GUI) for saving figures

    asyncio.run(main()) # Run the async main function