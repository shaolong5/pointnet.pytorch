"""
kinematic_model.py - Hierarchical Articulation Model (Improved)
微调版本，提高训练稳定性和兼容性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

# ======================== Base Modules ========================
class PointNetEncoder(nn.Module):
    """PointNet Encoder to extract local and global features from point cloud."""
    def __init__(self, input_dim: int = 3, local_feat_dim: int = 256, global_feat_dim: int = 1024):
        super().__init__()
        # Local feature extraction (pointwise MLP)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, local_feat_dim, 1)
        # Global feature extraction (pointwise MLP + max pool)
        self.conv4 = nn.Conv1d(local_feat_dim, 512, 1)
        self.conv5 = nn.Conv1d(512, global_feat_dim, 1)
        # Batch norm layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(local_feat_dim)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(global_feat_dim)
        self.local_feat_dim = local_feat_dim
        self.global_feat_dim = global_feat_dim
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, 3] input point cloud
        Returns: local_features [B, N, local_feat_dim], global_feature [B, global_feat_dim]
        """
        B, N, _ = x.shape
        # Transpose to [B, 3, N] for conv1d
        x = x.transpose(1, 2)
        
        # Local feature (per point)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        local_feat = F.relu(self.bn3(self.conv3(x)))    # [B, local_feat_dim, N]
        
        # Global feature (aggregate)
        x = F.relu(self.bn4(self.conv4(local_feat)))
        x = F.relu(self.bn5(self.conv5(x)))             # [B, global_feat_dim, N]
        global_feat = torch.max(x, dim=2)[0]            # [B, global_feat_dim]
        
        # Transpose local_feat back to [B, N, C]
        local_feat = local_feat.transpose(1, 2)         # [B, N, local_feat_dim]
        
        return local_feat, global_feat

class HierarchicalFeatureFusion(nn.Module):
    """Hierarchical feature fusion module to combine local and global features."""
    def __init__(self, local_dim: int, global_dim: int, output_dim: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(local_dim + global_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加轻微dropout防止过拟合
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
        # Initialize weights
        for m in self.fusion:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        """
        local_feat: [B, N, local_dim]
        global_feat: [B, global_dim]
        Returns: fused_features [B, N, output_dim]
        """
        B, N, _ = local_feat.shape
        # Expand global feature to per-point
        global_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)   # [B, N, global_dim]
        # Concatenate local and global features
        combined = torch.cat([local_feat, global_expanded], dim=-1)    # [B, N, local_dim+global_dim]
        # Reshape to apply MLP (BatchNorm1d expects [*, C] flat input)
        combined_flat = combined.reshape(B * N, -1)
        output_flat = self.fusion(combined_flat)                       # [B*N, output_dim]
        output = output_flat.reshape(B, N, -1)                         # [B, N, output_dim]
        return output

# ======================== Main Model ========================
class HierarchicalArticulateKinematicModel(nn.Module):
    """
    Hierarchical Kinematic Model for articulated objects.
    
    针对论文目标优化：
    1. 识别和分割关节部件（门、抽屉、开关等）
    2. 学习每个部件的运动学模型（类型、轴、原点、范围）
    3. 分割相关的交互元素（把手、旋钮、按钮等）
    """
    def __init__(self, 
                 num_large_parts: int = 11, 
                 num_interactive_elements: int = 12,
                 input_dim: int = 3, 
                 hidden_dim: int = 512,
                 use_dropout: bool = True):
        """
        Args:
            num_large_parts: 大型关节部件类别数（如门、抽屉等）
            num_interactive_elements: 交互元素类别数（如把手、按钮等）
            input_dim: 输入点云维度（通常为3）
            hidden_dim: 隐藏层维度
            use_dropout: 是否使用dropout
        """
        super().__init__()
        self.num_large_parts = num_large_parts
        self.num_interactive_elements = num_interactive_elements
        self.use_dropout = use_dropout
        
        # Feature extraction
        self.encoder = PointNetEncoder(input_dim, local_feat_dim=256, global_feat_dim=1024)
        
        # Feature fusion (combine local and global features)
        self.feature_fusion = HierarchicalFeatureFusion(256, 1024, hidden_dim)
        
        # Stage 1: Large articulated part segmentation head
        # (+1 for background class)
        dropout_rate = 0.3 if use_dropout else 0.0
        self.large_part_seg = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_large_parts + 1)  # +1 for background
        )
        
        # Kinematic parameter heads (关节运动学参数)
        self.joint_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # 减少dropout以提高精度
            nn.Linear(128, 3)  # 0=fixed, 1=revolute, 2=prismatic
        )
        
        self.joint_axis_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3)   # 3D axis vector
        )
        
        self.joint_origin_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 3)   # 3D origin coordinates
        )
        
        self.joint_range_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)   # range parameters
        )
        
        # Stage 2: Interactive element segmentation (交互元素分割)
        # 条件于大部件分割结果
        self.interactive_seg = nn.Sequential(
            nn.Linear(hidden_dim + (num_large_parts + 1), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_interactive_elements + 1)  # +1 for background
        )
        
        # Interactive association head (关联大部件)
        self.interactive_association = nn.Sequential(
            nn.Linear(hidden_dim + (num_large_parts + 1), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_large_parts + 1)
        )
        
        # Stage 3: Hierarchical relationship predictor
        self.hierarchy_predictor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_large_parts * num_interactive_elements)
        )
        
        # Confidence head (置信度预测)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize all weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all modules"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _reshape_for_bn(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten [B, N, C] to [B*N, C] for batch norm"""
        B, N, C = x.shape
        return x.reshape(B * N, C)

    def _reshape_back(self, x: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Reshape [B*N, C] back to [B, N, C]"""
        return x.reshape(B, N, -1)

    def forward(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            xyz: [B, N, 3] input point cloud
            
        Returns:
            Dictionary containing all predictions:
            - large_part_logits: [B, N, num_large_parts+1] 大部件分割
            - jt_logits: [B, N, 3] 关节类型
            - axis: [B, N, 3] 关节轴（归一化）
            - origin: [B, N, 3] 关节原点
            - range: [B, N, 2] 运动范围
            - interactive_logits: [B, N, num_interactive_elements+1] 交互元素分割
            - interactive_assoc: [B, N, num_large_parts+1] 交互元素关联
            - hierarchy_matrix: [B, num_large_parts, num_interactive_elements] 层次关系
            - confidence: [B, N] 置信度
        """
        B, N, _ = xyz.shape
        
        # Feature extraction
        local_feat, global_feat = self.encoder(xyz)
        
        # Feature fusion
        fused_features = self.feature_fusion(local_feat, global_feat)
        
        # Flatten for fully-connected heads
        fused_flat = self._reshape_for_bn(fused_features)

        # Stage 1: Large part segmentation
        large_part_logits = self.large_part_seg(fused_flat)
        large_part_logits = self._reshape_back(large_part_logits, B, N)

        # Kinematic parameter predictions
        joint_type_logits = self.joint_type_head(fused_flat)
        joint_type_logits = self._reshape_back(joint_type_logits, B, N)

        joint_axis_raw = self.joint_axis_head(fused_flat)
        joint_axis_raw = self._reshape_back(joint_axis_raw, B, N)
        joint_axis = F.normalize(joint_axis_raw, dim=-1, eps=1e-8)

        joint_origin = self.joint_origin_head(fused_flat)
        joint_origin = self._reshape_back(joint_origin, B, N)

        joint_range_raw = self.joint_range_head(fused_flat)
        joint_range_raw = self._reshape_back(joint_range_raw, B, N)
        
        # Convert range to [min, max] format
        center = joint_range_raw[..., 0:1]
        half_width = F.softplus(joint_range_raw[..., 1:2])
        joint_range = torch.cat([center - half_width, center + half_width], dim=-1)

        # Stage 2: Interactive element prediction (conditional)
        large_part_probs = F.softmax(large_part_logits.detach(), dim=-1)
        interactive_input = torch.cat([fused_features, large_part_probs], dim=-1)
        interactive_flat = self._reshape_for_bn(interactive_input)

        interactive_logits = self.interactive_seg(interactive_flat)
        interactive_logits = self._reshape_back(interactive_logits, B, N)

        interactive_assoc = self.interactive_association(interactive_flat)
        interactive_assoc = self._reshape_back(interactive_assoc, B, N)

        # Stage 3: Hierarchical relationship
        hierarchy_matrix = self.hierarchy_predictor(global_feat)
        hierarchy_matrix = hierarchy_matrix.view(B, self.num_large_parts, self.num_interactive_elements)
        hierarchy_matrix = torch.sigmoid(hierarchy_matrix)

        # Confidence prediction
        confidence_flat = self.confidence_head(fused_flat)
        confidence = self._reshape_back(confidence_flat, B, N).squeeze(-1)

        # Clean outputs (remove NaN/Inf)
        def clean(t: torch.Tensor, name: str = "") -> torch.Tensor:
            result = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
            if not torch.isfinite(result).all():
                print(f"[WARN] {name} contains non-finite values after cleanup")
            return result

        return {
            "large_part_logits": clean(large_part_logits, "large_part_logits"),
            "jt_logits": clean(joint_type_logits, "jt_logits"),
            "axis": clean(joint_axis, "axis"),
            "origin": clean(joint_origin, "origin"),
            "range": clean(joint_range, "range"),
            "interactive_logits": clean(interactive_logits, "interactive_logits"),
            "interactive_assoc": clean(interactive_assoc, "interactive_assoc"),
            "hierarchy_matrix": clean(hierarchy_matrix, "hierarchy_matrix"),
            "confidence": clean(confidence, "confidence")
        }

# ======================== Simplified Model (Optional) ========================
class SimpleArticulateModel(nn.Module):
    """
    简化版本的模型，用于快速原型测试
    只包含核心功能：部件分割、关节类型、交互元素
    """
    def __init__(self, num_large_parts: int = 11, num_interactive: int = 2):
        super().__init__()
        self.num_large_parts = num_large_parts
        self.num_interactive = num_interactive
        
        # Simple PointNet backbone
        self.feat1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.feat2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        # Output heads
        self.large_part_head = nn.Linear(512, num_large_parts + 1)
        self.joint_type_head = nn.Linear(512, 3)
        self.interactive_head = nn.Linear(512, num_interactive)
        
    def forward(self, xyz: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = xyz.shape
        
        # Feature extraction
        x = self.feat1(xyz)
        x = self.feat2(x)
        
        # Global pooling
        global_feat = torch.max(x, dim=1, keepdim=True)[0]
        feat = global_feat.expand(-1, N, -1) + x
        
        # Predictions
        return {
            "large_part_logits": self.large_part_head(feat),
            "jt_logits": self.joint_type_head(feat),
            "interactive_logits": self.interactive_head(feat),
            # Dummy outputs for compatibility
            "axis": F.normalize(torch.randn(B, N, 3, device=xyz.device), dim=-1),
            "origin": torch.zeros(B, N, 3, device=xyz.device),
            "range": torch.zeros(B, N, 2, device=xyz.device),
            "interactive_assoc": torch.zeros(B, N, self.num_large_parts + 1, device=xyz.device),
            "hierarchy_matrix": torch.zeros(B, self.num_large_parts, self.num_interactive, device=xyz.device),
            "confidence": torch.ones(B, N, device=xyz.device) * 0.5
        }