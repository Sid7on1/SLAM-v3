# SLAM V3 - EXACT SPECIFICATION IMPLEMENTATION (NO RoPE, COMPLETE COMPLIANCE)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math

# === CONFIG ===
EMBED_DIM = 1024  # BERT Large embedding dimension
SEQ_LEN = 100     # Sequence length for rotation example
NUM_HEADS = 16    # Multi-head attention heads
NUM_EXPERTS = 8   # Number of MoE experts
FF_DIM = 4096     # Feed-forward hidden dimension
VOCAB_SIZE = 30522  # BERT vocabulary size
TOP_K_EXPERTS = 2   # Top-K experts to use
NUM_ROTATIONS = 4   # Number of Q-segment rotations per EF cycle
MAX_EF_CYCLES = 3   # Maximum EF cycles
LOAD_BALANCE_WEIGHT = 0.01
CONVERGENCE_THRESHOLD = 1e-4
DROPOUT_RATE = 0.1
FUSION_METHOD = "average"  # Options: "average", "concat", "gated"

# === Expert Module ===
class Expert(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()  # GELU everywhere as per spec
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def __call__(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x)

# === MoE Multi-Head Self-Attention (EXACT SLAM V3 SPEC) ===
class MoEMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts, top_k=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.head_dim = embed_dim // num_heads
        
        # Expert-based QKV routing (MoE MHSA)
        self.q_experts = [nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)]
        self.k_experts = [nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)]
        self.v_experts = [nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)]
        
        # Gating network for expert selection
        self.gate = nn.Linear(embed_dim, num_experts)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def compute_load_balance_loss(self, gate_logits):
        """Compute load balancing loss"""
        gate_probs = mx.softmax(gate_logits, axis=-1)
        expert_usage = mx.mean(gate_probs, axis=(0, 1))
        load_balance_loss = LOAD_BALANCE_WEIGHT * mx.sum(expert_usage ** 2)
        return load_balance_loss
    
    def __call__(self, x, fixed_k=None, fixed_v=None, past_key_values=None, use_cache=False):
        batch_size, seq_len, _ = x.shape
        
        # Expert-based QKV routing
        gate_logits = self.gate(x)
        
        # Correct top-k extraction
        if hasattr(mx, 'topk') and len(mx.topk(gate_logits, self.top_k, axis=-1)) == 2:
            top_k_values, top_k_indices = mx.topk(gate_logits, self.top_k, axis=-1)
        else:
            top_k_values = mx.topk(gate_logits, self.top_k, axis=-1)
            top_k_indices = mx.argpartition(gate_logits, -self.top_k, axis=-1)[..., -self.top_k:]
        
        top_k_weights = mx.softmax(top_k_values, axis=-1)
        load_balance_loss = self.compute_load_balance_loss(gate_logits)
        
        # Compute Q using selected experts
        q_outputs = []
        for expert_idx in range(self.num_experts):
            expert_mask = mx.any(top_k_indices == expert_idx, axis=-1)
            if mx.sum(expert_mask) > 0:
                q_expert = self.q_experts[expert_idx](x)
                q_outputs.append(q_expert)
        
        if q_outputs:
            q_combined = mx.mean(mx.stack(q_outputs, axis=0), axis=0)
        else:
            q_combined = mx.zeros((batch_size, seq_len, self.embed_dim))
        
        # CRITICAL: Use fixed K/V if provided (EF cycles), otherwise compute fresh K/V (Level 1 & 3)
        if fixed_k is not None and fixed_v is not None:
            k_combined = fixed_k
            v_combined = fixed_v
        else:
            # Compute K/V using selected experts
            k_outputs = []
            v_outputs = []
            
            for expert_idx in range(self.num_experts):
                expert_mask = mx.any(top_k_indices == expert_idx, axis=-1)
                if mx.sum(expert_mask) > 0:
                    k_expert = self.k_experts[expert_idx](x)
                    v_expert = self.v_experts[expert_idx](x)
                    k_outputs.append(k_expert)
                    v_outputs.append(v_expert)
            
            if k_outputs:
                k_combined = mx.mean(mx.stack(k_outputs, axis=0), axis=0)
                v_combined = mx.mean(mx.stack(v_outputs, axis=0), axis=0)
            else:
                k_combined = mx.zeros((batch_size, seq_len, self.embed_dim))
                v_combined = mx.zeros((batch_size, seq_len, self.embed_dim))
        
        # KV Caching for autoregressive decoding
        if use_cache and past_key_values is not None:
            past_k, past_v = past_key_values
            k_combined = mx.concatenate([past_k, k_combined], axis=1)
            v_combined = mx.concatenate([past_v, v_combined], axis=1)
        
        # Reshape for multi-head attention
        q = q_combined.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k_combined.reshape(batch_size, k_combined.shape[1], self.num_heads, self.head_dim)
        v = v_combined.reshape(batch_size, v_combined.shape[1], self.num_heads, self.head_dim)
        
        # CRITICAL: Standard multi-head attention (NO POSITIONAL ENCODING)
        # SLAM v3 relies on structure + Q-Bias to learn positions, NOT RoPE
        q = q.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(0, 2, 1, 3)  # [batch, heads, kv_seq_len, head_dim]
        v = v.transpose(0, 2, 1, 3)  # [batch, heads, kv_seq_len, head_dim]
        
        scores = mx.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        
        # Causal mask
        kv_seq_len = k.shape[2]
        mask = mx.triu(mx.ones((seq_len, kv_seq_len)), k=1)
        scores = mx.where(mask == 1, -mx.inf, scores)
        
        attn_weights = mx.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = mx.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Prepare cache for next iteration
        present_key_values = (k_combined, v_combined) if use_cache else None
        
        return out, k_combined, v_combined, load_balance_loss, present_key_values

# === MoE FFN with Load Balancing ===
class MoEFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.embed_dim = embed_dim
        
        # Individual expert networks
        self.experts = [Expert(embed_dim, hidden_dim) for _ in range(num_experts)]
        
        # Gating network
        self.gate = nn.Linear(embed_dim, num_experts)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def compute_load_balance_loss(self, gate_logits):
        """Compute load balancing loss"""
        gate_probs = mx.softmax(gate_logits, axis=-1)
        expert_usage = mx.mean(gate_probs, axis=(0, 1))
        load_balance_loss = LOAD_BALANCE_WEIGHT * mx.sum(expert_usage ** 2)
        return load_balance_loss
    
    def __call__(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute gating scores
        gate_logits = self.gate(x)
        
        # Correct top-k extraction
        if hasattr(mx, 'topk') and len(mx.topk(gate_logits, self.top_k, axis=-1)) == 2:
            top_k_values, top_k_indices = mx.topk(gate_logits, self.top_k, axis=-1)
        else:
            top_k_values = mx.topk(gate_logits, self.top_k, axis=-1)
            top_k_indices = mx.argpartition(gate_logits, -self.top_k, axis=-1)[..., -self.top_k:]
        
        top_k_probs = mx.softmax(top_k_values, axis=-1)
        load_balance_loss = self.compute_load_balance_loss(gate_logits)
        
        # Sparse computation
        output = mx.zeros_like(x)
        for expert_idx in range(self.num_experts):
            expert_mask = mx.any(top_k_indices == expert_idx, axis=-1)
            if mx.sum(expert_mask) > 0:
                expert_weights = mx.zeros((batch_size, seq_len))
                for k in range(self.top_k):
                    k_mask = (top_k_indices[..., k] == expert_idx)
                    expert_weights += k_mask * top_k_probs[..., k]
                
                expert_output = self.experts[expert_idx](x)
                output += expert_output * mx.expand_dims(expert_weights, axis=-1)
        
        output = self.dropout(output)
        return output, load_balance_loss

# === LEVEL 1: Global Initialization ===
class Level1Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts):
        super().__init__()
        # MoE-MHSA (not basic attention)
        self.attn = MoEMultiHeadSelfAttention(embed_dim, num_heads, num_experts, top_k=TOP_K_EXPERTS)
        self.ffn = MoEFFN(embed_dim, ff_dim, num_experts, top_k=TOP_K_EXPERTS)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def __call__(self, x):
        # MoE Self-Attention
        attn_out, k_cache, v_cache, attn_load_loss, _ = self.attn(self.norm1(x))
        x = x + attn_out
        
        # MoE FFN
        ffn_out, ffn_load_loss = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        total_load_loss = attn_load_loss + ffn_load_loss
        return x, k_cache, v_cache, total_load_loss

# === EF Block with Q-Bias and FIXED K/V ===
class EFBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, block_id, q_bias_value):
        super().__init__()
        self.block_id = block_id
        self.block_name = ['A', 'B', 'C', 'D'][block_id]
        
        # MoE-MHSA (not basic attention)
        self.attn = MoEMultiHeadSelfAttention(embed_dim, num_heads, num_experts, top_k=TOP_K_EXPERTS)
        self.ffn = MoEFFN(embed_dim, ff_dim, num_experts, top_k=TOP_K_EXPERTS)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # CRITICAL: Learnable Q-bias (block-anchored, FIXED per block)
        # Block A always gets Q_bias_A, Block B gets Q_bias_B, etc.
        self.q_bias = mx.random.normal((embed_dim,)) * q_bias_value
        print(f"EF Block {self.block_name}: Q-bias initialized with scale {q_bias_value}")
    
    def __call__(self, q_segment, fixed_k, fixed_v):
        # CRITICAL: Apply learnable Q-bias ONLY to Q (not K/V)
        q_with_bias = q_segment + self.q_bias
        
        # CRITICAL: Use EXACT fixed K/V from Level 1 (NO recomputation, FULL context)
        # This is the key insight: Q segments attend to FULL K/V context
        # This allows cross-segment attention and is what makes EF cycles powerful
        
        # MoE Attention with FULL fixed K/V from Level 1 (no segmentation)
        attn_out, _, _, attn_load_loss, _ = self.attn(self.norm1(q_with_bias), fixed_k, fixed_v)
        q_segment = q_segment + attn_out
        
        # MoE FFN
        ffn_out, ffn_load_loss = self.ffn(self.norm2(q_segment))
        q_segment = q_segment + ffn_out
        
        total_load_loss = attn_load_loss + ffn_load_loss
        return q_segment, total_load_loss

# === Flexible Fusion Module ===
class FlexibleFusion(nn.Module):
    def __init__(self, embed_dim, fusion_method="average"):
        super().__init__()
        self.fusion_method = fusion_method
        self.embed_dim = embed_dim
        
        if fusion_method == "concat":
            self.fusion_proj = nn.Linear(embed_dim * 4, embed_dim)
        elif fusion_method == "gated":
            self.gate_proj = nn.Linear(embed_dim, 4)  # 4 blocks
            self.fusion_proj = nn.Linear(embed_dim, embed_dim)
    
    def __call__(self, block_outputs):
        """
        CRITICAL: Fuse 4 outputs per token from 4 rotations
        Args:
            block_outputs: List of 4 tensors from 4 rotations
        Returns:
            Fused tensor of shape [batch, seq_len, embed_dim]
        """
        if self.fusion_method == "average":
            # Simple average fusion
            return mx.mean(mx.stack(block_outputs, axis=0), axis=0)
        
        elif self.fusion_method == "concat":
            # Concatenate and project
            concatenated = mx.concatenate(block_outputs, axis=-1)
            return self.fusion_proj(concatenated)
        
        elif self.fusion_method == "gated":
            # Gated fusion with attention-like weights
            stacked = mx.stack(block_outputs, axis=0)  # [4, batch, seq_len, embed_dim]
            
            # Compute gate weights for each rotation
            query = mx.mean(block_outputs[0], axis=1, keepdims=True)  # [batch, 1, embed_dim]
            gate_logits = self.gate_proj(query)  # [batch, 1, 4]
            gate_weights = mx.softmax(gate_logits, axis=-1)  # [batch, 1, 4]
            
            # Apply gated fusion
            gate_weights = mx.expand_dims(gate_weights, axis=-1)  # [batch, 1, 4, 1]
            weighted = stacked * gate_weights.transpose(2, 0, 1, 3)  # [4, batch, seq_len, embed_dim]
            fused = mx.sum(weighted, axis=0)  # [batch, seq_len, embed_dim]
            
            return self.fusion_proj(fused)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

# === LEVEL 2: EF Cycles Implementation ===
class Level2EFCycles(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, fusion_method="average"):
        super().__init__()
        
        # 4 EF blocks with different Q-bias values
        # Block A → Structure, Block B → Numerics, Block C → Causality, Block D → Emotion
        q_bias_values = {
            'A': 0.1,    # Structure
            'B': 0.2,    # Numerics  
            'C': 0.3,    # Causality
            'D': 0.4     # Emotion/General
        }
        
        self.blocks = {
            'A': EFBlock(embed_dim, num_heads, ff_dim, num_experts, 0, q_bias_values['A']),
            'B': EFBlock(embed_dim, num_heads, ff_dim, num_experts, 1, q_bias_values['B']),
            'C': EFBlock(embed_dim, num_heads, ff_dim, num_experts, 2, q_bias_values['C']),
            'D': EFBlock(embed_dim, num_heads, ff_dim, num_experts, 3, q_bias_values['D'])
        }
        
        # Flexible fusion for 4 outputs per token
        self.fusion = FlexibleFusion(embed_dim, fusion_method)
        self._is_training = True
    
    def get_q_segments(self, x, rotation_round):
        """
        Q-segment rotation logic
        Each block gets different Q segments per rotation
        """
        batch_size, seq_len, embed_dim = x.shape
        segment_size = seq_len // NUM_ROTATIONS  # 25 tokens per segment
        
        # Rotation pattern as per SLAM v3 spec
        # Round 1: A(1-25), B(26-50), C(51-75), D(76-100)  
        # Round 2: A(26-50), B(51-75), C(76-100), D(1-25)
        # Round 3: A(51-75), B(76-100), C(1-25), D(26-50)
        # Round 4: A(76-100), B(1-25), C(26-50), D(51-75)
        
        rotation_map = {
            0: {'A': 0, 'B': 1, 'C': 2, 'D': 3},  # Round 1
            1: {'A': 1, 'B': 2, 'C': 3, 'D': 0},  # Round 2  
            2: {'A': 2, 'B': 3, 'C': 0, 'D': 1},  # Round 3
            3: {'A': 3, 'B': 0, 'C': 1, 'D': 2}   # Round 4
        }
        
        segments = {}
        for block_name in ['A', 'B', 'C', 'D']:
            segment_idx = rotation_map[rotation_round][block_name]
            start_idx = segment_idx * segment_size
            end_idx = min(start_idx + segment_size, seq_len)
            segments[block_name] = x[:, start_idx:end_idx, :]
            
        return segments
    
    def __call__(self, h_level1, k_l1, v_l1):
        """
        CRITICAL: EF Cycles with proper K/V reuse and Q rotation
        Args:
            h_level1: Output from Level 1
            k_l1: FIXED K from Level 1 (reused across ALL EF cycles)
            v_l1: FIXED V from Level 1 (reused across ALL EF cycles)
        """
        print(f"\\n=== LEVEL 2: EF Cycles Implementation ===")
        print(f"CRITICAL: K/V FIXED from Level 1: K{k_l1.shape}, V{v_l1.shape}")
        print(f"CRITICAL: K/V will NOT be recomputed in any EF cycle")
        
        # CRITICAL: K/V stay FIXED from Level 1 across ALL EF cycles
        fixed_k = k_l1
        fixed_v = v_l1
        
        current_q = h_level1  # Initial Q from Level 1
        total_load_loss = 0.0
        
        for cycle in range(MAX_EF_CYCLES):
            print(f"\\n--- EF Cycle {cycle + 1} ---")
            print(f"CRITICAL: Using SAME fixed K/V from Level 1 (no recomputation)")
            
            # Each EF cycle has 4 rotations
            # Each rotation processes ALL 4 blocks in parallel with different Q segments
            # Then we collect 4 outputs per token (one from each rotation) and fuse them
            
            all_rotation_outputs = []  # Collect outputs from all rotations
            cycle_load_loss = 0.0
            
            for rotation in range(NUM_ROTATIONS):
                print(f"  Rotation {rotation + 1}:")
                
                # Get Q segments for this rotation
                q_segments = self.get_q_segments(current_q, rotation)
                
                # Process each block with its assigned Q segment
                rotation_block_outputs = []
                
                for block_name in ['A', 'B', 'C', 'D']:
                    q_seg = q_segments[block_name]
                    print(f"    Block {block_name}: Q segment {q_seg.shape}, Q-bias applied")
                    
                    # CRITICAL: Use FULL fixed K/V from Level 1 (no recomputation, no segmentation)
                    block_out, block_load_loss = self.blocks[block_name](q_seg, fixed_k, fixed_v)
                    rotation_block_outputs.append(block_out)
                    cycle_load_loss += block_load_loss
                
                # Reconstruct full sequence from segments for this rotation
                # Reorder segments back to original sequence order
                ordered_outputs = [None] * NUM_ROTATIONS
                rotation_map = {
                    0: {'A': 0, 'B': 1, 'C': 2, 'D': 3},
                    1: {'A': 1, 'B': 2, 'C': 3, 'D': 0},
                    2: {'A': 2, 'B': 3, 'C': 0, 'D': 1},
                    3: {'A': 3, 'B': 0, 'C': 1, 'D': 2}
                }
                
                for i, block_name in enumerate(['A', 'B', 'C', 'D']):
                    original_pos = rotation_map[rotation][block_name]
                    ordered_outputs[original_pos] = rotation_block_outputs[i]
                
                rotation_output = mx.concatenate(ordered_outputs, axis=1)
                all_rotation_outputs.append(rotation_output)
            
            # CRITICAL: Now we have 4 full-sequence outputs (one per rotation)
            # Each token now has 4 different representations from 4 rotations
            # This is the "4 outputs per token" that need to be fused
            print(f"  CRITICAL: Fusing 4 rotation outputs (4 outputs per token)")
            print(f"  Each rotation output shape: {[out.shape for out in all_rotation_outputs]}")
            print(f"  Fusion method: {self.fusion.fusion_method}")
            
            fused_output = self.fusion(all_rotation_outputs)
            
            # Check convergence
            output_diff = mx.mean(mx.abs(fused_output - current_q))
            print(f"  Cycle {cycle + 1} convergence: {output_diff:.6f}")
            
            if not self._is_training and output_diff < CONVERGENCE_THRESHOLD:
                print(f"  Converged at cycle {cycle + 1}")
                break
                
            # CRITICAL: Pass ONLY fused Q to next EF cycle (K/V stay fixed)
            current_q = fused_output
            total_load_loss += cycle_load_loss
        
        print(f"EF Cycles completed. Final Q shape: {current_q.shape}")
        print(f"CRITICAL: K/V remained FIXED throughout all EF cycles")
        return current_q, total_load_loss

# === LEVEL 3: Final Context Stitching ===
class Level3FinalStitching(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts):
        super().__init__()
        # MoE-MHSA (not basic attention)
        self.attn = MoEMultiHeadSelfAttention(embed_dim, num_heads, num_experts, top_k=TOP_K_EXPERTS)
        self.ffn = MoEFFN(embed_dim, ff_dim, num_experts, top_k=TOP_K_EXPERTS)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def __call__(self, h_ef):
        # Fresh Q/K/V computation for final stitching
        attn_out, k_enc, v_enc, attn_load_loss, _ = self.attn(self.norm1(h_ef))
        h_final = h_ef + attn_out
        
        # Final FFN
        ffn_out, ffn_load_loss = self.ffn(self.norm2(h_final))
        h_enc_final = h_final + ffn_out  # Proper naming as per spec
        
        total_load_loss = attn_load_loss + ffn_load_loss
        return h_enc_final, k_enc, v_enc, total_load_loss

# === Shared Decoder ===
class SharedDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        # MoE-MHSA (not basic attention)
        self.self_attn = MoEMultiHeadSelfAttention(embed_dim, num_heads, num_experts, top_k=TOP_K_EXPERTS)
        self.cross_attn = MoEMultiHeadSelfAttention(embed_dim, num_heads, num_experts, top_k=TOP_K_EXPERTS)
        self.ffn = MoEFFN(embed_dim, ff_dim, num_experts, top_k=TOP_K_EXPERTS)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def __call__(self, decoder_input, k_enc, v_enc, past_key_values=None, use_cache=False):
        total_load_loss = 0.0
        
        # Causal self-attention with KV caching
        self_past_kv = past_key_values[0] if past_key_values else None
        self_attn_out, _, _, self_load_loss, self_present_kv = self.self_attn(
            self.norm1(decoder_input), 
            past_key_values=self_past_kv, 
            use_cache=use_cache
        )
        decoder_input = decoder_input + self_attn_out
        total_load_loss += self_load_loss
        
        # Cross-attention with encoder
        cross_attn_out, _, _, cross_load_loss, _ = self.cross_attn(
            self.norm2(decoder_input), k_enc, v_enc
        )
        decoder_input = decoder_input + cross_attn_out
        total_load_loss += cross_load_loss
        
        # FFN
        ffn_out, ffn_load_loss = self.ffn(self.norm3(decoder_input))
        decoder_input = decoder_input + ffn_out
        total_load_loss += ffn_load_loss
        
        # Output projection
        logits = self.output_proj(decoder_input)
        
        # Prepare cache for next iteration
        present_key_values = [self_present_kv] if use_cache else None
        
        return logits, total_load_loss, present_key_values

# === Complete SLAM V3 Model (EXACT SPEC MATCH) ===
class SLAMV3Model(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, vocab_size, fusion_method="average"):
        super().__init__()
        self.embed_dim = embed_dim
        
        # CRITICAL: NO POSITIONAL ENCODING (NO RoPE, NO LEARNED POSITIONS)
        # SLAM v3 relies on structure + Q-Bias to learn positions
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.token_type_embedding = nn.Embedding(2, embed_dim)
        
        # 3-Level Architecture
        self.level1 = Level1Encoder(embed_dim, num_heads, ff_dim, num_experts)
        self.level2 = Level2EFCycles(embed_dim, num_heads, ff_dim, num_experts, fusion_method)
        self.level3 = Level3FinalStitching(embed_dim, num_heads, ff_dim, num_experts)
        
        # Shared decoder
        self.decoder = SharedDecoder(embed_dim, num_heads, ff_dim, num_experts, vocab_size)
        
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def __call__(self, input_ids, decoder_input_ids=None, token_type_ids=None, 
                 past_key_values=None, use_cache=False):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Token type embeddings
        if token_type_ids is None:
            token_type_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)
        token_type_embeds = self.token_type_embedding(token_type_ids)
        
        # CRITICAL: NO POSITIONAL ENCODING - only token + token_type embeddings
        x = token_embeds + token_type_embeds
        x = self.dropout(x)
        
        print(f"Input embeddings shape: {x.shape}")
        print(f"CRITICAL: NO positional encoding used (NO RoPE, NO learned positions)")
        print(f"CRITICAL: Position learning relies on structure + Q-Bias")
        
        # LEVEL 1: Global Initialization
        print(f"\\n=== LEVEL 1: Global Initialization (MoE-MHSA) ===")
        h_level1, k_l1, v_l1, level1_load_loss = self.level1(x)
        print(f"Level 1 output shape: {h_level1.shape}")
        print(f"K/V cache shapes: {k_l1.shape}, {v_l1.shape}")
        
        # LEVEL 2: EF Cycles (K/V fixed from Level 1)
        h_ef, level2_load_loss = self.level2(h_level1, k_l1, v_l1)
        
        # LEVEL 3: Final Stitching
        print(f"\\n=== LEVEL 3: Final Context Stitching (MoE-MHSA) ===")
        h_enc_final, k_enc, v_enc, level3_load_loss = self.level3(h_ef)
        print(f"Level 3 output shape (H_enc_final): {h_enc_final.shape}")
        print(f"CRITICAL: Decoder prep - K_enc: {k_enc.shape}, V_enc: {v_enc.shape}")
        print(f"✅ K_enc and V_enc cached for decoder cross-attention")
        
        # Aggregate all load balance losses
        total_load_loss = level1_load_loss + level2_load_loss + level3_load_loss
        
        # Decoder (if decoder input provided)
        if decoder_input_ids is not None:
            decoder_token_embeds = self.token_embedding(decoder_input_ids)
            decoder_embeddings = self.dropout(decoder_token_embeds)
            
            logits, decoder_load_loss, present_key_values = self.decoder(
                decoder_embeddings, k_enc, v_enc, past_key_values, use_cache
            )
            total_load_loss += decoder_load_loss
            
            return h_enc_final, logits, total_load_loss, present_key_values
        
        return h_enc_final, total_load_loss

# === Test Function ===
def test_slam_v3():
    
    # Create model
    model = SLAMV3Model(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, 
        ff_dim=FF_DIM,
        num_experts=NUM_EXPERTS,
        vocab_size=VOCAB_SIZE,
        fusion_method=FUSION_METHOD
    )
    
    # Create test data
    batch_size = 1
    seq_len = SEQ_LEN
    
    # Random input token IDs
    input_ids = mx.random.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    decoder_input_ids = mx.random.randint(0, VOCAB_SIZE, (batch_size, seq_len // 2))
    token_type_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)  # Single segment
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Decoder input IDs shape: {decoder_input_ids.shape}")
    print(f"Token type IDs shape: {token_type_ids.shape}")
    
    # Forward pass
    h_enc_final, decoder_logits, total_load_loss, _ = model(
        input_ids, decoder_input_ids, token_type_ids, use_cache=False
    )
    
    print(f"\\n=== Final Results ===")
    print(f"H_enc_final shape: {h_enc_final.shape}")
    print(f"Decoder logits shape: {decoder_logits.shape}")
    print(f"H_enc_final mean: {mx.mean(h_enc_final):.4f}")
    print(f"H_enc_final std: {mx.std(h_enc_final):.4f}")
    print(f"Total load balance loss: {total_load_loss:.6f}")
    
    # Calculate score
    score = float(mx.mean(mx.abs(h_enc_final))) * 1000
    print(f"\\nSLAM V3 Score: {score:.2f}")
    

    
    return score

if __name__ == "__main__":
    test_slam_v3()