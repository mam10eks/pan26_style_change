#import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
#from transformers import AutoTokenizer, AutoModelForMaskedLM
#import jsonlines
#from torch.utils.data import Dataset
#from torch.utils.data import DataLoader
from transformers import AutoModel
#from transformers import DataCollatorWithPadding
#from tqdm.auto import tqdm
#import bokeh.io
#bokeh.io.output_notebook()

#BASE_MODEL_PATH = "microsoft/mdeberta-v3-base"

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#base_mlm = AutoModel.from_pretrained(BASE_MODEL_PATH,torch_dtype=torch.float32, output_hidden_states=True,)#, num_labels=NUM_LABELS

#base_mlm.config.output_hidden_states = True

#merged_encoder = merged_encoder.to(device)

# ──────────────────────────────────────────────────────────────────────────────
# Gromov-Wasserstein Loss
# ──────────────────────────────────────────────────────────────────────────────


def gromov_wasserstein_loss(src_cls, tgt_cls, n_samples=256, n_iter=5):
	"""
	Entropic Gromov-Wasserstein via Sinkhorn iterations.
	No labels needed — aligns geometric structure.
	"""
	# Subsample for efficiency
	idx_s = torch.randperm(len(src_cls))[:n_samples]
	idx_t = torch.randperm(len(tgt_cls))[:n_samples]
	X = src_cls[idx_s]  # (n, d)
	Y = tgt_cls[idx_t]  # (n, d)

	# Intra-space distance matrices
	C_src = torch.cdist(X, X, p=2).pow(2)  # (n, n)
	C_tgt = torch.cdist(Y, Y, p=2).pow(2)  # (n, n)

	# Normalize
	C_src = C_src / C_src.max()
	C_tgt = C_tgt / C_tgt.max()

	# Uniform marginals
	n = n_samples
	T = torch.ones(n, n, device=X.device) / (n * n)  # init transport plan

	# Sinkhorn-GW iterations
	eps = 0.1
	for _ in range(n_iter):
		# GW cost given current T
		M = (
			C_src.pow(2) @ T @ torch.ones(n, n, device=X.device) +
			torch.ones(n, n, device=X.device) @ T @ C_tgt.pow(2) -
			2 * C_src @ T @ C_tgt.T
		)
		# Sinkhorn step
		log_T = (-M / eps)
		log_T = log_T - torch.logsumexp(log_T, dim=1, keepdim=True)
		log_T = log_T - torch.logsumexp(log_T, dim=0, keepdim=True)
		T = log_T.exp()

	return (M * T).sum()

# ──────────────────────────────────────────────────────────────────────────────
# Cosine Similarity Loss
# ──────────────────────────────────────────────────────────────────────────────



class CosineSimilarityLoss(nn.Module):
	def __init__(self, temperature: float = 0.07):
		super().__init__()
		self.temperature = temperature
		self.loss_fct = nn.BCEWithLogitsLoss()

	def forward(self, emb_a, emb_b, labels):
		cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)
		# Scale into BCE-friendly logit range
		# cos_sim / temp maps [-1,1] → [-14, 14] at temp=0.07
		scaled_sim = cos_sim / self.temperature
		return self.loss_fct(scaled_sim, labels.float())


# ──────────────────────────────────────────────────────────────────────────────
# Cosine Margin Loss
# ──────────────────────────────────────────────────────────────────────────────


class CosineMarginLoss(nn.Module):
	def __init__(self, margin: float = 0.3):
		super().__init__()
		self.margin = margin

	def forward(self, emb_a, emb_b, labels):
		sim    = F.cosine_similarity(emb_a, emb_b, dim=-1)
		labels = labels.float()
		# Gradient is constant ±1 — no saturation, no explosion
		pos_loss = labels       * (1 - sim)                      # same author → push sim to 1
		neg_loss = (1 - labels) * F.relu(sim - self.margin)      # diff author → push sim below margin
		return (pos_loss + neg_loss).mean()

# ──────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ──────────────────────────────────────────────────────────────────────────────

class GradientReversalFunction(Function):
	"""
	Forward pass: identity
	Backward pass: multiplies gradient by -alpha
	This forces the encoder to learn language-invariant features by
	trying to fool the language discriminator.
	"""
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.clone()

	@staticmethod
	def backward(ctx, grad_output):
		return -ctx.alpha * grad_output, None  # reverse + scale gradient


class GradientReversal(nn.Module):
	def __init__(self, alpha: float = 1.0):
		super().__init__()
		self.alpha = alpha

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return GradientReversalFunction.apply(x, self.alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Language Discriminator
# ──────────────────────────────────────────────────────────────────────────────

class LanguageDiscriminator(nn.Module):
	"""
	Tries to predict the language of the input from the fused embedding.
	The gradient reversal layer ensures the encoder is trained to
	make this task as hard as possible → language-invariant embeddings.
	"""
	def __init__(self, hidden_size: int, num_languages: int, dropout: float = 0.1):
		super().__init__()
		self.reversal = GradientReversal()          # alpha set dynamically in forward
		self.classifier = nn.Sequential(
			nn.Linear(hidden_size, hidden_size // 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size // 2, num_languages),
		)
		self.loss_fct = nn.CrossEntropyLoss()

	def forward(
		self,
		embeddings: torch.Tensor,       # (B, hidden_size) fused CLS embeddings
		language_ids: torch.Tensor,     # (B,) integer language labels
		alpha: float = 1.0,             # reversal strength — schedule this during training
	) -> tuple[torch.Tensor, torch.Tensor]:
		self.reversal.alpha = alpha
		reversed_emb = self.reversal(embeddings)            # gradient reversed here
		logits = self.classifier(reversed_emb)              # (B, num_languages)
		loss = self.loss_fct(logits, language_ids)
		return loss, logits


# ──────────────────────────────────────────────────────────────────────────────
# Alpha schedule (standard DANN schedule)
# ──────────────────────────────────────────────────────────────────────────────

def get_reversal_alpha(current_step: int, total_steps: int, gamma: float = 10.0) -> float:
	"""
	Gradually increases alpha from 0 → 1 over training.
	Starts gentle (small alpha) so the adversary doesn't destabilise
	early training, then ramps up as the encoder stabilises.

	Usage:
		alpha = get_reversal_alpha(global_step, total_steps)
	"""
	p = current_step / total_steps
	return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))) - 1.0)




class MultiLayerGatedFusion(nn.Module):
	"""
	Fuses CLS + mean embeddings from multiple layers.
	"""
	def __init__(self, hidden_size, num_layers, dropout=0.2):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = nn.Dropout(dropout)
		
		# Total input size: (CLS + MEAN) * num_layers
		fusion_input = hidden_size * num_layers

		# Gating layer
		self.gate = nn.Linear(fusion_input, hidden_size)

		# Final projection to hidden_size
		self.proj = nn.Sequential(
			nn.Linear(hidden_size, hidden_size),
			nn.GELU(),
			nn.LayerNorm(hidden_size)
		)

	def forward(self, cls_list):
		"""
		cls_list: list of tensors from different layers [(batch, hidden), ...]
		mean_list: same shape
		"""
		concat = torch.cat([cls for cls in cls_list], dim=-1)
		concat = self.dropout(concat)
		gate_values = torch.sigmoid(self.gate(concat))  # (batch, hidden)

		# Weighted average of all layers using the gate
		# Sum of gated layer representations
		combined_cls = sum(cls_list) / len(cls_list)
		#combined_mean = sum(mean_list) / len(mean_list)

		fused = gate_values * combined_cls #+ (1 - gate_values) * combined_mean

		return self.proj(fused)  # (batch, hidden)

class DebertaV3ClassifierGW(nn.Module):
	"""
	DebertaV3Classifier with Gromov-Wasserstein domain alignment
	replacing the adversarial language discriminator.

	GW loss aligns the geometric structure of source and target domain
	CLS token distributions without requiring language labels or
	gradient reversal. It minimises the discrepancy between pairwise
	distance matrices of the two embedding clouds, forcing the encoder
	to produce structurally similar representations across languages.

	Key difference from adversarial training:
		Adversarial: needs language_ids per sample, gradient reversal
		GW:          needs a target domain batch alongside source batch,
					 no labels, no reversal — purely geometric alignment

	Training loop change:
		For each source batch, supply a target batch from an unlabelled
		target language loader (e.g. Polish test data without labels).
		GW loss is only computed when tgt_input_ids is provided.

	Args:
		lambda_gw      : weight of GW loss term
		gw_n_samples   : subsampling size for GW (keep ≤ batch size)
		gw_n_iter      : Sinkhorn iterations
		gw_eps         : entropic regularisation strength
		gw_pooling     : "cls" | "mean" | "both" — what to align
	"""

	def __init__(
		self,
		backbone_model,
		num_labels: int = 1,
		dropout: float = 0.1,
		num_unfrozen: int = 2,
		use_cls: bool = True,
		# ── Segment similarity (unchanged) ────────────────────────────
		use_segment_sim: bool = True,
		lambda_sim: float = 0.1,
		sim_margin: float = 0.3,
		# ── GW domain alignment ───────────────────────────────────────
		lambda_gw: float = 0.1,
		gw_n_samples: int = 128,        # subsample — keep ≤ batch size
		gw_n_iter: int = 5,
		gw_eps: float = 0.1,
		gw_pooling: str = "cls",        # "cls" | "mean" | "both"
		pos_weight: float = 1.0,
	):
		super().__init__()
		assert gw_pooling in ("cls", "mean", "both")

		self.num_unfrozen    = num_unfrozen
		self.use_segment_sim = use_segment_sim
		self.lambda_sim      = lambda_sim
		self.lambda_gw       = lambda_gw
		self.gw_n_samples    = gw_n_samples
		self.gw_n_iter       = gw_n_iter
		self.gw_eps          = gw_eps
		self.gw_pooling      = gw_pooling

		self.backbone: nn.Module = backbone_model
		self.backbone.config.output_hidden_states = True
		hidden_size = self.backbone.config.hidden_size

		self.fusion = MultiLayerGatedFusion(
			hidden_size=hidden_size,
			num_layers=num_unfrozen,
			dropout=dropout,
		)
		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(hidden_size, hidden_size),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, num_labels),
		)
		self.pos_weight = torch.Tensor([pos_weight]).float()
		self.loss_fct    = nn.BCEWithLogitsLoss()
		self.sim_loss_fn = CosineMarginLoss(margin=sim_margin)

	# ── Helpers (reused from previous version) ─────────────────────────

	def _get_special_ids(self) -> set[int]:
		cfg = self.backbone.config
		ids = set()
		for attr in ["bos_token_id", "eos_token_id", "pad_token_id"]:
			val = getattr(cfg, attr, None)
			if val is not None:
				ids.add(val)
		ids.update({0, 1, 2})
		return ids

	def _content_mean(self, layer_hidden, input_ids, attention_mask, special_ids):
		special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
		for sid in special_ids:
			special_mask |= (input_ids == sid)
		content_mask   = attention_mask.bool() & ~special_mask
		content_mask_f = content_mask.unsqueeze(-1).float()
		sum_h  = (layer_hidden * content_mask_f).sum(dim=1)
		count  = content_mask_f.sum(dim=1).clamp(min=1e-9)
		return sum_h / count

	def _extract_segment_means(self, last_hidden, input_ids, attention_mask, special_ids):
		B, seq_len, H = last_hidden.shape
		sep_id    = getattr(self.backbone.config, "eos_token_id", 2)
		is_sep    = (input_ids == sep_id)
		first_sep = is_sep.float().argmax(dim=1)
		positions = (
			torch.arange(seq_len, device=input_ids.device)
				 .unsqueeze(0).expand(B, -1)
		)
		special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
		for sid in special_ids:
			special_mask |= (input_ids == sid)
		seg_a_mask = (
			(positions > 0) &
			(positions < first_sep.unsqueeze(1)) &
			~special_mask & attention_mask.bool()
		)
		seg_b_mask = (
			(positions > first_sep.unsqueeze(1)) &
			~special_mask & attention_mask.bool()
		)
		def masked_mean(hidden, mask):
			mask_f = mask.unsqueeze(-1).float()
			return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
		return masked_mean(last_hidden, seg_a_mask), \
			   masked_mean(last_hidden, seg_b_mask)

	# ── GW loss ────────────────────────────────────────────────────────

	def _gromov_wasserstein_loss(
		self,
		src_emb: torch.Tensor,          # (B_src, H)
		tgt_emb: torch.Tensor,          # (B_tgt, H)
	) -> torch.Tensor:
		"""
		Entropic Gromov-Wasserstein via Sinkhorn.
		Aligns pairwise distance structure of src and tgt clouds.
		Gradients flow back through src_emb — pushes encoder toward
		producing geometrically similar representations across domains.
		"""
		n = min(self.gw_n_samples, src_emb.size(0), tgt_emb.size(0))

		idx_s = torch.randperm(src_emb.size(0), device=src_emb.device)[:n]
		idx_t = torch.randperm(tgt_emb.size(0), device=tgt_emb.device)[:n]

		X = src_emb[idx_s]              # (n, H) — gradients flow here
		Y = tgt_emb[idx_t].detach()     # (n, H) — target is fixed reference

		C_src = torch.cdist(X, X, p=2).pow(2)
		C_tgt = torch.cdist(Y, Y, p=2).pow(2)

		# Normalise to [0, 1] for numerical stability
		C_src = C_src / (C_src.max() + 1e-9)
		C_tgt = C_tgt / (C_tgt.max() + 1e-9)

		ones  = torch.ones(n, n, device=X.device)
		T     = ones / (n * n)          # uniform init transport plan

		for _ in range(self.gw_n_iter):
			M = (
				C_src.pow(2) @ T @ ones +
				ones @ T @ C_tgt.pow(2) -
				2 * C_src @ T @ C_tgt.T
			)
			log_T = -M / self.gw_eps
			log_T = log_T - torch.logsumexp(log_T, dim=1, keepdim=True)
			log_T = log_T - torch.logsumexp(log_T, dim=0, keepdim=True)
			T     = log_T.exp().detach()   # detach T — only M carries grad

		return (M * T).sum()

	def _pool_for_gw(
		self,
		last_layers: list[torch.Tensor],
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		special_ids: set[int],
	) -> torch.Tensor:
		"""Produces the embedding used for GW alignment per the gw_pooling mode."""
		# Average pooled representation across unfrozen layers
		cls_list  = [layer[:, 0] for layer in last_layers]
		mean_list = [
			self._content_mean(layer, input_ids, attention_mask, special_ids)
			for layer in last_layers
		]
		cls_mean  = torch.stack(cls_list,  dim=0).mean(dim=0)   # (B, H)
		mean_mean = torch.stack(mean_list, dim=0).mean(dim=0)   # (B, H)

		if self.gw_pooling == "cls":
			return cls_mean
		elif self.gw_pooling == "mean":
			return mean_mean
		else:  # both — concatenate
			return torch.cat([cls_mean, mean_mean], dim=-1)     # (B, 2H)

	# ── Encoder ────────────────────────────────────────────────────────

	def _encode(self, input_ids, attention_mask):
		"""Shared encoder — returns last_layers, cls_list, special_ids."""
		outputs       = self.backbone(input_ids=input_ids,
									  attention_mask=attention_mask)
		hidden_states = outputs.hidden_states
		last_layers   = hidden_states[-self.num_unfrozen:]
		cls_list      = [layer[:, 0] for layer in last_layers]
		special_ids   = self._get_special_ids()
		return last_layers, cls_list, special_ids

	# ── Forward ────────────────────────────────────────────────────────

	def forward(
		self,
		input_ids: torch.Tensor,
		attention_mask: torch.Tensor,
		labels: torch.Tensor | None = None,
		# Target domain batch — unlabelled, different language
		tgt_input_ids: torch.Tensor | None = None,
		tgt_attention_mask: torch.Tensor | None = None,
	) -> dict:

		# ── Source domain encoding ─────────────────────────────────────
		last_layers, cls_list, special_ids = self._encode(
			input_ids, attention_mask
		)

		# ── Segment similarity loss (source batch only, unchanged) ─────
		sim_loss   = None
		mean_a_out = None
		mean_b_out = None
		sim_out    = None

		if self.use_segment_sim and labels is not None:
			seg_sim_losses = []
			for layer in last_layers:
				mean_a, mean_b = self._extract_segment_means(
					layer, input_ids, attention_mask, special_ids
				)
				mean_a_n = F.normalize(mean_a, p=2, dim=-1)
				mean_b_n = F.normalize(mean_b, p=2, dim=-1)
				seg_sim_losses.append(
					self.sim_loss_fn(mean_a_n, mean_b_n, labels.float())
				)
			sim_loss   = torch.stack(seg_sim_losses).mean()
			mean_a_out = mean_a_n
			mean_b_out = mean_b_n
			sim_out    = F.cosine_similarity(mean_a_n, mean_b_n, dim=-1)

		# ── GW domain alignment loss ───────────────────────────────────
		gw_loss = None

		if tgt_input_ids is not None:
			# Encode target domain — no labels, no gradient to classifier
			tgt_last_layers, _, tgt_special_ids = self._encode(
				tgt_input_ids, tgt_attention_mask
			)
			src_emb = self._pool_for_gw(
				last_layers, input_ids, attention_mask, special_ids
			)
			tgt_emb = self._pool_for_gw(
				tgt_last_layers, tgt_input_ids, tgt_attention_mask,
				tgt_special_ids
			)
			gw_loss = self._gromov_wasserstein_loss(src_emb, tgt_emb)

		# ── Classification via CLS ─────────────────────────────────────
		fused_embedding = self.fusion(cls_list)
		logits          = self.classifier(fused_embedding)

		# ── Combined loss ──────────────────────────────────────────────
		loss = 0
		if labels is not None:
			bce_loss = self.loss_fct(logits, labels.float().unsqueeze(1))
			loss += bce_loss
			if sim_loss is not None:
				loss = loss + self.lambda_sim * sim_loss
			if gw_loss is not None:
				gw_loss_normalised = gw_loss * (bce_loss.detach() / (gw_loss.detach() + 1e-9))
				loss = loss + self.lambda_gw * gw_loss_normalised

		return {
			"loss":           loss,
			"logits":         logits,
			"cls_embeddings": fused_embedding,
			"sim_loss":       sim_loss,
			"gw_loss":        gw_loss,
			"mean_a":         mean_a_out,
			"mean_b":         mean_b_out,
			"similarity":     sim_out,
		}

#import torch
#import torch.nn as nn
#import torch.nn.functional as F


# Default bucket boundaries (in content tokens).
DEFAULT_LENGTH_BUCKETS = (16, 24, 32, 48, 64, 96, 128, 192, 256)


class AVClassifier(nn.Module):
	"""
	Classifier for Authorship Verification with the backbone being Roberta,
	XLM-R, or DeBERTa models.

	Head-side conditioning supports four optional feature groups, each toggleable:

	  - length:   [log(len_a+1), log(len_b+1), |log diff|, bucket_emb(min(len_a, len_b))]
	  - source:   nn.Embedding(num_sources, source_emb_dim)(source_ids)
	  - domain:   nn.Embedding(num_domains, domain_emb_dim)(domain_ids)
	  - language: nn.Embedding(num_languages, language_emb_dim)(language_ids)

	All enabled features are concatenated to the fused CLS embedding before
	the classifier MLP. Each embedding is initialized small (std=0.02) so the
	head starts close to the unconditioned baseline.

	Missing IDs at runtime are handled by zero-vector fallback — useful at
	eval time when a source/language is not present in the training maps.
	"""

	def __init__(
		self,
		backbone_model,
		num_labels: int = 1,
		dropout: float = 0.1,
		num_unfrozen: int = 2,
		use_cls: bool = True,
		# ── Segment similarity (unchanged) ────────────────────────────
		use_segment_sim: bool = True,
		lambda_sim: float = 0.1,
		sim_margin: float = 0.3,
		# ── Length conditioning ───────────────────────────────────────
		use_length_features: bool = True,
		length_bucket_boundaries: tuple = DEFAULT_LENGTH_BUCKETS,
		length_bucket_emb_dim: int = 32,
		# ── Source / domain / language conditioning (new) ─────────────
		use_source_features: bool = True,
		num_sources: int = 13,
		source_emb_dim: int = 32,
		use_domain_features: bool = True,
		num_domains: int = 5,
		domain_emb_dim: int = 16,
		use_language_features: bool = True,
		num_languages: int = 8,
		language_emb_dim: int = 16,
	):
		super().__init__()

		self.num_unfrozen          = num_unfrozen
		self.use_segment_sim       = use_segment_sim
		self.lambda_sim            = lambda_sim
		self.use_length_features   = use_length_features
		self.use_source_features   = use_source_features
		self.use_domain_features   = use_domain_features
		self.use_language_features = use_language_features

		self.backbone: nn.Module = backbone_model
		self.backbone.config.output_hidden_states = True
		hidden_size = self.backbone.config.hidden_size

		self.fusion = MultiLayerGatedFusion(
			hidden_size=hidden_size,
			num_layers=num_unfrozen,
			dropout=dropout,
		)

		# ── Length conditioning ───────────────────────────────────────
		if self.use_length_features:
			self.register_buffer(
				"length_bucket_boundaries",
				torch.tensor(list(length_bucket_boundaries), dtype=torch.long),
				persistent=False,
			)
			num_buckets = len(length_bucket_boundaries) + 1
			self.length_bucket_emb = nn.Embedding(num_buckets, length_bucket_emb_dim)
			nn.init.normal_(self.length_bucket_emb.weight, mean=0.0, std=0.02)
			length_feat_dim = 3 + length_bucket_emb_dim
		else:
			length_feat_dim = 0

		# ── Source conditioning ───────────────────────────────────────
		if self.use_source_features:
			self.source_emb = nn.Embedding(num_sources, source_emb_dim)
			nn.init.normal_(self.source_emb.weight, mean=0.0, std=0.02)
			source_feat_dim = source_emb_dim
		else:
			source_feat_dim = 0

		# ── Domain conditioning ───────────────────────────────────────
		if self.use_domain_features:
			self.domain_emb = nn.Embedding(num_domains, domain_emb_dim)
			nn.init.normal_(self.domain_emb.weight, mean=0.0, std=0.02)
			domain_feat_dim = domain_emb_dim
		else:
			domain_feat_dim = 0

		# ── Language conditioning ─────────────────────────────────────
		if self.use_language_features:
			self.language_emb = nn.Embedding(num_languages, language_emb_dim)
			nn.init.normal_(self.language_emb.weight, mean=0.0, std=0.02)
			language_feat_dim = language_emb_dim
		else:
			language_feat_dim = 0

		# ── Classifier head ───────────────────────────────────────────
		self.total_cond_dim = (
			length_feat_dim + source_feat_dim + domain_feat_dim + language_feat_dim
		)
		classifier_in_dim = hidden_size + self.total_cond_dim

		self.classifier = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(classifier_in_dim, hidden_size),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, num_labels),
		)
		self.loss_fct    = nn.BCEWithLogitsLoss()
		self.sim_loss_fn = CosineMarginLoss(margin=sim_margin)

	# ── Helpers ────────────────────────────────────────────────────────

	def _get_special_ids(self) -> set[int]:
		cfg = self.backbone.config
		ids = set()
		for attr in (
			"bos_token_id", "eos_token_id", "pad_token_id",
			"sep_token_id", "cls_token_id", "unk_token_id", "mask_token_id",
		):
			val = getattr(cfg, attr, None)
			if val is not None:
				ids.add(val)
		return ids

	def _content_mean(self, layer_hidden, input_ids, attention_mask, special_ids):
		special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
		for sid in special_ids:
			special_mask |= (input_ids == sid)
		content_mask   = attention_mask.bool() & ~special_mask
		content_mask_f = content_mask.unsqueeze(-1).float()
		sum_h  = (layer_hidden * content_mask_f).sum(dim=1)
		count  = content_mask_f.sum(dim=1).clamp(min=1e-9)
		return sum_h / count

	def _extract_segment_masks(self, input_ids, attention_mask, special_ids):
			"""Return (seg_a_mask, seg_b_mask) bool tensors over content tokens."""
			B, seq_len = input_ids.shape

			# Resolve SEP id robustly across model families (RoBERTa uses eos_token_id=2,
			# DeBERTa uses sep_token_id=2 with eos_token_id=None).
			sep_id = None
			for attr in ("sep_token_id", "eos_token_id"):
				val = getattr(self.backbone.config, attr, None)
				if val is not None:
					sep_id = val
					break
			if sep_id is None:
				sep_id = 2

			is_sep    = (input_ids == sep_id)
			first_sep = is_sep.float().argmax(dim=1)
			positions = (
				torch.arange(seq_len, device=input_ids.device)
					.unsqueeze(0).expand(B, -1)
			)
			special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
			for sid in special_ids:
				special_mask |= (input_ids == sid)
			seg_a_mask = (
				(positions > 0) &
				(positions < first_sep.unsqueeze(1)) &
				~special_mask & attention_mask.bool()
			)
			seg_b_mask = (
				(positions > first_sep.unsqueeze(1)) &
				~special_mask & attention_mask.bool()
			)
			return seg_a_mask, seg_b_mask

	def _extract_segment_means(self, last_hidden, input_ids, attention_mask, special_ids):
		seg_a_mask, seg_b_mask = self._extract_segment_masks(
			input_ids, attention_mask, special_ids
		)
		def masked_mean(hidden, mask):
			mask_f = mask.unsqueeze(-1).float()
			return (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
		return masked_mean(last_hidden, seg_a_mask), \
			   masked_mean(last_hidden, seg_b_mask)

	def _compute_length_features(self, input_ids, attention_mask, special_ids):
		"""Compute length-conditioning features. Returns (B, 3 + bucket_emb_dim)."""
		seg_a_mask, seg_b_mask = self._extract_segment_masks(
			input_ids, attention_mask, special_ids
		)
		len_a = seg_a_mask.sum(dim=1).float()
		len_b = seg_b_mask.sum(dim=1).float()

		log_len_a = torch.log1p(len_a)
		log_len_b = torch.log1p(len_b)
		abs_diff  = torch.abs(log_len_a - log_len_b)

		min_len    = torch.minimum(len_a, len_b).long()
		bucket_idx = torch.bucketize(min_len, self.length_bucket_boundaries)
		bucket_emb = self.length_bucket_emb(bucket_idx)

		scalars = torch.stack([log_len_a, log_len_b, abs_diff], dim=-1)
		return torch.cat([scalars, bucket_emb], dim=-1)

	def _maybe_embed(self, ids, embedding_module, batch_size, device, dtype):
		"""
		Look up an embedding for `ids` if provided; otherwise return zeros.
		Lets the model handle eval-time examples where source/language is unknown.
		"""
		if ids is not None:
			return embedding_module(ids)
		return torch.zeros(
			batch_size,
			embedding_module.embedding_dim,
			device=device,
			dtype=dtype,
		)

	# ── Encoder ────────────────────────────────────────────────────────

	def _encode(self, input_ids, attention_mask):
		outputs       = self.backbone(input_ids=input_ids,
									  attention_mask=attention_mask)
		hidden_states = outputs.hidden_states
		last_layers   = hidden_states[-self.num_unfrozen:]
		cls_list      = [layer[:, 0] for layer in last_layers]
		special_ids   = self._get_special_ids()
		return last_layers, cls_list, special_ids

	# ── Forward ────────────────────────────────────────────────────────

	def forward(
		self,
		input_ids:      torch.Tensor,
		attention_mask: torch.Tensor,
		labels:         torch.Tensor | None = None,
		source_ids:     torch.Tensor | None = None,
		domain_ids:     torch.Tensor | None = None,
		language_ids:   torch.Tensor | None = None,
		**kwargs,  # absorb any extra fields the collator passes through
	) -> dict:

		# ── Encode ─────────────────────────────────────────────────────
		last_layers, cls_list, special_ids = self._encode(
			input_ids, attention_mask
		)

		# ── Segment similarity loss (unchanged) ────────────────────────
		sim_loss   = None
		mean_a_out = None
		mean_b_out = None
		sim_out    = None

		if self.use_segment_sim and labels is not None:
			seg_sim_losses = []
			for layer in last_layers:
				mean_a, mean_b = self._extract_segment_means(
					layer, input_ids, attention_mask, special_ids
				)
				mean_a_n = F.normalize(mean_a, p=2, dim=-1)
				mean_b_n = F.normalize(mean_b, p=2, dim=-1)
				seg_sim_losses.append(
					self.sim_loss_fn(mean_a_n, mean_b_n, labels.float())
				)
			sim_loss   = torch.stack(seg_sim_losses).mean()
			mean_a_out = mean_a_n
			mean_b_out = mean_b_n
			sim_out    = F.cosine_similarity(mean_a_n, mean_b_n, dim=-1)

		# ── Fuse CLS across unfrozen layers ────────────────────────────
		fused_embedding = self.fusion(cls_list)
		B      = fused_embedding.shape[0]
		device = fused_embedding.device
		dtype  = fused_embedding.dtype

		# ── Assemble conditioning features ─────────────────────────────
		cond_feats = []

		if self.use_length_features:
			cond_feats.append(self._compute_length_features(
				input_ids, attention_mask, special_ids
			))

		if self.use_source_features:
			cond_feats.append(self._maybe_embed(
				source_ids, self.source_emb, B, device, dtype
			))

		if self.use_domain_features:
			cond_feats.append(self._maybe_embed(
				domain_ids, self.domain_emb, B, device, dtype
			))

		if self.use_language_features:
			cond_feats.append(self._maybe_embed(
				language_ids, self.language_emb, B, device, dtype
			))

		if cond_feats:
			classifier_input = torch.cat([fused_embedding] + cond_feats, dim=-1)
		else:
			classifier_input = fused_embedding

		logits = self.classifier(classifier_input)

		# ── Combined loss ──────────────────────────────────────────────
		loss = 0
		if labels is not None:
			bce_loss = self.loss_fct(logits, labels.float().unsqueeze(1))
			loss += bce_loss
			if sim_loss is not None:
				loss = loss + self.lambda_sim * sim_loss

		return {
			"loss":           loss,
			"logits":         logits,
			"cls_embeddings": fused_embedding,
			"sim_loss":       sim_loss,
			"mean_a":         mean_a_out,
			"mean_b":         mean_b_out,
			"similarity":     sim_out,
		}
#num_unfrozen = 2

# model = AVClassifier(
# 					backbone_model=base_mlm, 
# 					num_labels=1, dropout=0.1, 
# 					#num_languages=len(LANGUAGE_TO_ID),
# 					use_segment_sim=False,
# 					num_unfrozen=num_unfrozen, #use_cls=True,
# 					# ── Length conditioning ───────────────────────────────────────
# 					use_length_features = True,
# 					length_bucket_boundaries = DEFAULT_LENGTH_BUCKETS,
# 					length_bucket_emb_dim = 32,
# 					# ── Source / domain / language conditioning (new) ─────────────
# 					use_source_features = True,
# 					num_sources = 13,
# 					source_emb_dim = 32,
# 					use_domain_features = True,
# 					num_domains = 5,
# 					domain_emb_dim = 32,
# 					use_language_features = True,
# 					num_languages = 9,
# 					language_emb_dim = 32,
# 					).to(device)

# model.eval()