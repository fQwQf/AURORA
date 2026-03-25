"""AURORA Configuration Dataclass.

Replaces the proliferation of OursV4-V15 with a single, clean config object.
Every ablation variant is a specific instantiation of AURORAConfig.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AURORAConfig:
    """Unified configuration for all AURORA variants and ablations.

    Canonical AURORA (V14 equivalent):
        use_etf_anchors=True
        use_uncertainty_weighting=True
        use_dynamic_attenuation=True
        use_stability_reg=True
        aggregation='feature_ensemble'

    Ablation variants (toggle ONE flag at a time from canonical):
        A1 - No ETF (random anchors):     use_etf_anchors=False
        A2 - No uncertainty weighting:    use_uncertainty_weighting=False, fixed_lambda=5.0
        A3 - No dynamic attenuation:      use_dynamic_attenuation=False
        A4 - No stability reg:            use_stability_reg=False
        A5 - Feature collapse (ablation): force_feature_alignment=True
        A6 - No alignment at all:         fixed_lambda=0.0
    """

    # ── Alignment target ─────────────────────────────────────────────────────
    use_etf_anchors: bool = True
    """If True, use ETF anchors. If False, use random fixed anchors (ablation A1)."""

    # ── Lambda scheduling ────────────────────────────────────────────────────
    use_uncertainty_weighting: bool = True
    """If True, λ is learned via homoscedastic uncertainty (meta-annealing).
    If False, fixed_lambda is used directly (ablation A2)."""

    fixed_lambda: float = 0.0
    """Used when use_uncertainty_weighting=False.
    Set to 0.0 to disable alignment entirely (ablation A6)."""

    use_adaptive_lambda_init: bool = True
    """If True, initialize σ params from data entropy (from V9 logic).
    Only applies when use_uncertainty_weighting=True."""

    lambda_min: float = 0.1
    lambda_max_init: float = 15.0  # For adaptive lambda initialization

    # ── Dynamic task attenuation (Meta-Annealing) ────────────────────────────
    use_dynamic_attenuation: bool = True
    """Cosine schedule on log(σ_align²), causing λ_eff to decay over training.
    Ablation A3: set to False."""

    # ── Stability regularization ─────────────────────────────────────────────
    use_stability_reg: bool = True
    """ReLU hinge: γ·ReLU(λ_eff - λ_max). Prevents λ explosion.
    Ablation A4: set to False (gamma_reg=0)."""

    gamma_reg: float = 1e-5
    """Regularization strength γ. Only effective when use_stability_reg=True."""

    lambda_max: float = 50.0
    """Upper threshold for stability regularization."""

    # ── Gradient routing ─────────────────────────────────────────────────────
    force_feature_alignment: bool = False
    """Ablation A5: align features directly to anchors instead of prototypes.
    This is the 'Feature Collapse' ablation - proves prototype layer is necessary."""

    # ── Sigma learning rate ──────────────────────────────────────────────────
    sigma_lr: float = 0.005
    """Learning rate for σ parameters. Separate from backbone lr."""

    # ── Server aggregation strategy ──────────────────────────────────────────
    aggregation: str = 'feature_ensemble'
    """Options: 'feature_ensemble', 'fedavg'
    feature_ensemble: WEnsembleFeature + global proto nearest-neighbor
    fedavg: parameter averaging (for AURORAFedAvg ablation)
    """

    # ── Logging & analysis ───────────────────────────────────────────────────
    log_metrics: bool = True
    """If True, log rich per-round metrics: λ trajectory, proto alignment,
    feature geometry. Used to generate analysis plots."""

    # ── Named variant (for logging/paper tables) ─────────────────────────────
    variant_name: str = 'AURORA'

    @classmethod
    def canonical(cls, gamma_reg: float = 1e-5, lambda_max: float = 50.0,
                  sigma_lr: float = 0.005) -> 'AURORAConfig':
        """Full AURORA as described in the paper."""
        return cls(
            use_etf_anchors=True,
            use_uncertainty_weighting=True,
            use_adaptive_lambda_init=True,
            use_dynamic_attenuation=True,
            use_stability_reg=True,
            gamma_reg=gamma_reg,
            lambda_max=lambda_max,
            sigma_lr=sigma_lr,
            aggregation='feature_ensemble',
            variant_name='AURORA',
        )

    @classmethod
    def ablation_no_alignment(cls) -> 'AURORAConfig':
        """A6: No alignment. Pure FAFI baseline (SupCon + Proto losses only)."""
        return cls(
            use_uncertainty_weighting=False,
            fixed_lambda=0.0,
            variant_name='FAFI (No Align)',
        )

    @classmethod
    def ablation_fixed_lambda(cls, lam: float = 5.0) -> 'AURORAConfig':
        """A2: Fixed λ, no uncertainty weighting. Equivalent to FAFI+ETF+Fixed."""
        return cls(
            use_etf_anchors=True,
            use_uncertainty_weighting=False,
            fixed_lambda=lam,
            use_dynamic_attenuation=False,
            use_stability_reg=False,
            variant_name=f'Fixed-λ ({lam})',
        )

    @classmethod
    def ablation_no_etf(cls, gamma_reg: float = 1e-5,
                        lambda_max: float = 50.0) -> 'AURORAConfig':
        """A1: Random anchors instead of ETF."""
        return cls(
            use_etf_anchors=False,
            use_uncertainty_weighting=True,
            use_adaptive_lambda_init=True,
            use_dynamic_attenuation=True,
            use_stability_reg=True,
            gamma_reg=gamma_reg,
            lambda_max=lambda_max,
            variant_name='AURORA w/o ETF',
        )

    @classmethod
    def ablation_no_attenuation(cls, gamma_reg: float = 1e-5,
                                lambda_max: float = 50.0) -> 'AURORAConfig':
        """A3: No dynamic attenuation (constant uncertainty weighting)."""
        return cls(
            use_etf_anchors=True,
            use_uncertainty_weighting=True,
            use_adaptive_lambda_init=True,
            use_dynamic_attenuation=False,
            use_stability_reg=True,
            gamma_reg=gamma_reg,
            lambda_max=lambda_max,
            variant_name='AURORA w/o Attenuation',
        )

    @classmethod
    def ablation_no_stability(cls) -> 'AURORAConfig':
        """A4: No stability regularization."""
        return cls(
            use_etf_anchors=True,
            use_uncertainty_weighting=True,
            use_adaptive_lambda_init=True,
            use_dynamic_attenuation=True,
            use_stability_reg=False,
            gamma_reg=0.0,
            variant_name='AURORA w/o StabReg',
        )

    @classmethod
    def ablation_feature_collapse(cls) -> 'AURORAConfig':
        """A5: Align features directly (bypasses prototype layer).
        This is the proper Feature Collapse ablation."""
        return cls(
            use_etf_anchors=True,
            use_uncertainty_weighting=True,
            use_adaptive_lambda_init=True,
            use_dynamic_attenuation=True,
            use_stability_reg=True,
            force_feature_alignment=True,
            variant_name='AURORA-FeatureAlign (Collapse)',
        )

    @classmethod
    def aurora_fedavg(cls, gamma_reg: float = 1e-5,
                     lambda_max: float = 50.0) -> 'AURORAConfig':
        """AURORA with FedAvg aggregation instead of feature ensemble."""
        return cls(
            use_etf_anchors=True,
            use_uncertainty_weighting=True,
            use_adaptive_lambda_init=True,
            use_dynamic_attenuation=True,
            use_stability_reg=True,
            gamma_reg=gamma_reg,
            lambda_max=lambda_max,
            aggregation='fedavg',
            variant_name='AURORA+FedAvg',
        )
