我的一个学生的最近的工作结果是这样的，有什么问题？

OpenReview.net
Search OpenReview...
Notifications
Activity
Tasks
Jizhou Tong 
back arrowBack to Author Console
AURORA: Autonomous Regularization for One-shot Representation Alignment
Download PDF
Jizhou Tong, Bin Yang, Wenke Huang, Mang Ye 
24 Jan 2026 (modified: 10 Feb 2026)
ICML 2026 Conference Submission
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
CC BY 4.0
Verify Author List: I have double-checked the author list and understand that additions and removals will not be allowed after the abstract submission deadline.
TL;DR: AURORA addresses catastrophic failure in one-shot federated learning under extreme non-IID data by autonomously scheduling the balance between global geometric alignment and local adaptation.
Abstract:
One-shot Federated Learning (OFL) is crucial for scalable, privacy-preserving deployment, as it enables collaborative training with only a single communication round across bandwidth-limited clients. However, the one-shot constraint exacerbates non-IID heterogeneity, often leading to representation misalignment and severe cross-client model inconsistency. Prior OFL solutions (e.g., distillation/synthesis, improved aggregation, or client-side regularization) can benefit from geometric anchor alignment (e.g., Simplex ETF), but face two key obstacles: (I) Temporal Dichotomy: static constraints cannot smoothly transition from strong early alignment to late local adaptation; (II) Autonomous Scheduling: hand-designed schedules are costly to tune, while naive uncertainty weighting couples objectives and can introduce gradient interference and unstable optimization. To address these issues, we propose AURORA, an autonomous regularization framework with two modules: Meta-Annealing, which encodes a monotonic prior to transition from global alignment to local adaptation, and Gradient Decoupling, which separates uncertainty learning from backbone updates to enable stable, data-dependent schedule discovery. Experiments on CIFAR , SVHN and Tiny-ImageNet demonstrate state-of-the-art accuracy and improved stability under a single configuration. Code is available at https://anonymous.4open.science/r/AURORA-sf2d .

Supplementary Material:  zip
Primary Area: Social Aspects->Privacy
Keywords: Federated Learning, One-shot Learning, Non-IID Data, Model Alignment, Regularization, Meta-Learning, Neural Collapse
Ethics Agreement: I certify that all co-authors of this work have read and are committed to adhering to the Call for Papers, Author Instructions, Research Ethics, and Peer-review Ethics.
LLM Policy: This submission allows Policy B.
Reciprocal Reviewing Status: All of the qualified authors are serving as SACs, ACs, or in other organizing roles for ICML 2026, or are already listed as reciprocal reviewers on two submissions to ICML 2026.
Reciprocal Reviewing Author:  Bin Yang
Submission Number: 30402
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
4 / 4 replies shown
Withdrawal by Authors
Withdrawalby Authors24 Mar 2026, 23:00Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, Authors
Withdrawal Confirmation: I have read and agree with the venue's withdrawal policy on behalf of myself and my co-authors.
Official Review of Submission30402 by Reviewer Wu7C
Official Reviewby Reviewer Wu7C21 Mar 2026, 21:45 (modified: 24 Mar 2026, 22:44)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer Wu7CRevisions
Summary:
This paper proposes AURORA, a one-shot federated learning (OFL) method that aims to automatically balance the trade-off between local adaptation and global structural alignment. Specifically, it addresses the issue that the alignment strength (
) is both dataset-dependent and time-dependent. The method adopts uncertainty weighting with gradient decoupling of the uncertainty parameter 
 to automatically adjust 
 during training and stabilize optimization. As a result, 
 provides stronger alignment in the early stages of training and weaker alignment in later stages. Experimental results demonstrate that AURORA outperforms existing methods by a clear margin and is less sensitive to hyperparameters.

Strengths And Weaknesses:
Strength:

Overall, the paper provides interesting insights into the trade-off between local adaptation and global alignment and proposes a novel solution to address these challenges.

The experimental results are promising, demonstrating either significant performance gains compared to IntactOFL, FedETF, and AFL, or reduced manual effort, as evidenced by the comparison with FAFI+Anneal.

Weakness:

Although using predefined geometric anchors (Simplex ETF) ensures maximal separation, it does not capture semantic relationships between classes. For example, semantically similar classes may benefit from closer feature representations. This limitation may hinder generalization or transfer (fine-tuning) to additional classes.

The experimental study of OFL methods appears to focus only on training from scratch. Given that pre-trained weights have been shown to benefit federated learning [1], and considering recent advances in pre-trained or foundation models, the absence of this direction may render the results somewhat outdated and raises questions about their applicability in practical settings.

In addition to the primary objective, client-side optimization also involves optimizing a detached meta-objective for the uncertainty parameters 
. This may introduce additional computational overhead for clients, whose resources are typically assumed to be limited.

[1] 2023 ICLR On the Importance and Applicability of Pre-Training for Federated Learning

Soundness: 3: good
Presentation: 2: fair
Significance: 2: fair
Originality: 3: good
Key Questions For Authors:
Besides the weakness shown in the above section, please also see the following questions:

Q1: From the right column of Lines 399-404, the authors appear to focus primarily on the setting of training from scratch. However, as pretrained weights become more widely used, does AURORA remain competitive or outperform existing methods when initialized with pretrained models?

Q2: As the client side needs to optimize an additional meta-objective defined in Equation 8, does this make client training more computationally demanding? For example, what is the training time or number of iterations required by AURORA compared to other methods?

Limitations:
The dimensionality constraint of Simplex ETF, together with the use of a projector to map features to a higher-dimensional space, is evaluated only in settings with at most 100 or 200 classes (i.e., CIFAR-100 and Tiny-ImageNet). However, in practical scenarios, the number of classes can easily exceed 1,000 (e.g., ImageNet). It remains unclear whether Simplex ETF can still perform well at this scale.

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission30402 by Reviewer h6WK
Official Reviewby Reviewer h6WK11 Mar 2026, 16:06 (modified: 24 Mar 2026, 22:44)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer h6WKRevisions
Summary:
This paper addresses the challenge of extreme Non-IID data distributions in One-shot Federated Learning (OFL). The authors argue that fixed static alignment constraints can hinder model training in later stages, and propose Aurora, an autonomous regularization framework. Aurora aims to achieve dynamic, self-adaptive scheduling of alignment intensity through three key mechanisms: uncertainty weighting, gradient decoupling, and meta-annealing.

Strengths And Weaknesses:
Strengths:

The paper is well-written with intuitive figures (e.g., Figure 1 visualizes the Temporal Dichotomy), effectively conveying the core motivation.
The proposed method achieves adaptive representation alignment without manual intervention through gradient decoupling and meta-annealing, improving accuracy under extreme Non-IID conditions.
Weaknesses:

One of the core designs is to update prototypes only without backpropagating to the encoder, justified by preventing "Feature Collapse". However, in the ablation study (Table 21), the variant retaining gradients still achieves 43.73% accuracy, which is far above random chance or complete collapse.
Although OFL is a general problem, the evaluation is limited to small models (mainly ResNet-18). Given that current federated learning research widely involves cross-modal tasks and foundation models, validating this complex "soft alignment" strategy solely on small vision models makes it difficult to demonstrate its universality and importance in higher-dimensional spaces or different modalities (e.g., representation alignment in MLLM).
The paper relies heavily on existing technical components;.The use of Simplex ETF as global anchors to address heterogeneity alignment has already been proposed in prior works. From the perspective of deep learning methodology evolution, its novelty appears somewhat incremental.
Soundness: 2: fair
Presentation: 3: good
Significance: 1: poor
Originality: 2: fair
Key Questions For Authors:
1.The paper emphasizes that AURORA eliminates the need for manual scheduling and hyperparameter tuning. However, the sensitivity analysis in Table 13 reveals that altering the hyperparameter 
 (regularization strength) causes the accuracy on the SVHN dataset to plummet from 52.9% to a catastrophic 16.4%. Given the highly diverse and unpredictable data distributions in real-world federated learning, how can the authors guarantee that their default parameters are genuinely robust across various scenarios, rather than merely an empirical coincidence over-fitted to these specific datasets?

The ablation study examining "Feature Collapse" (Table 21) is strictly limited to the CIFAR-100 dataset. Furthermore, the "Ablation (No Detach)" variant, which directly propagates the alignment gradients to the encoder, still achieves an accuracy of 43.73%. This performance actually surpasses almost all the established baselines reported in the paper. Does this suggest that the "feature collapse" is somewhat exaggerated or misleading in the main text?

How would AURORA’s static geometric alignment mechanism adapt to the demands of anti-drift incremental fusion in federated scenarios? Specifically, the current evaluation is constrained to relatively small vision models; how viable is this framework when scaled up to Multi-modal Large Language Models (MLLMs), where the feature dimensions and category spaces are vastly larger and continuously evolving?

Limitations:
yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission30402 by Reviewer rQpo
Official Reviewby Reviewer rQpo10 Mar 2026, 20:50 (modified: 24 Mar 2026, 22:44)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer rQpoRevisions
Summary:
This paper proposes AURORA, a framework designed to address the "Temporal Dichotomy" in One-shot Federated Learning (OFL). By integrating Meta-Annealing and Gradient Decoupling, the framework enables an autonomous, data-driven alignment schedule. Additionally, it introduces Stability Regularization to prevent optimization collapse under extreme data skew.

Strengths And Weaknesses:
Strengths
Original and Accurate Problem Diagnosis: The paper correctly identifies that prior static alignment methods fail because they ignore temporal dynamics. Framing this as the "Temporal Dichotomy" is conceptually inspiring and distinct from general non-IID challenges.
Solid Technical Design: The motivation for Gradient Decoupling is rigorous. The authors theoretically demonstrate why standard uncertainty weighting destabilizes optimization, and the ablation studies provide compelling evidence for this design choice.
Clear Presentation: The three-layer architectural decomposition is intuitive. Figures 1 and 2 effectively and cleanly illustrate the Temporal Dichotomy and gradient flow boundaries.
Weaknesses
Disconnect Between Theory and Practice: The convergence analysis relies heavily on a quasi-static assumption (that losses vary slowly). However, in extreme non-IID settings—the paper's primary focus—this assumption is likely violated, meaning the theory does not fully capture the actual joint training dynamics.
Limited Experimental Scale and Dataset Representativeness: Core experiments use a very small client setting (
) and rely exclusively on Dirichlet-partitioned vision datasets. The lack of validation on datasets with natural federated structures (e.g., FEMNIST, medical imaging) weakens confidence in its real-world generalizability.
Questionable Hyperparameter Sensitivity Claims: Although the authors claim hyperparameters act merely as safety bounds , appendix data shows that tweaking sigma-lr causes a 3.2% performance swing, indicating it is highly performance-sensitive.
Marginal Novelty in Certain Components: While effective, mechanisms like the meta-annealing prior and stability regularization are fairly standard engineering safeguards, making them incremental refinements rather than independent innovations.
Soundness: 2: fair
Presentation: 2: fair
Significance: 1: poor
Originality: 3: good
Key Questions For Authors:
Under extreme non-IID scenarios where the quasi-static assumption is likely violated, how meaningful is the theoretical convergence guarantee for actual training dynamics?
How much of the reported performance gain is retained under standard aggregators (like FedAvg) compared to the specialized IFFI aggregator used in the main results?
Given the significant performance variance tied to sigma-lr (a 3.2% gap), does this parameter require dataset-specific fine-tuning in practical deployments?
How does AURORA perform on datasets with naturally occurring federated heterogeneity (e.g., FEMNIST) rather than Dirichlet simulations?
When scaling up the client count significantly (e.g., 
), does AURORA maintain its advantage in handling the Temporal Dichotomy and containing the lambda explosion risk?
Limitations:
Concerns Regarding AI-Generated Images: Several figures (e.g., Figure 1) exhibit visual characteristics consistent with AI-generated imagery, including decorative elements lacking precise technical grounding. The authors must ensure these visualizations rigorously and faithfully depict the described mechanisms rather than being purely aesthetic.
Inadequate Limitations Discussion: The current limitations section fails to fully address the method's potential reliance on the IFFI aggregator, the boundaries of the theoretical proofs regarding Assumption A2, and the evaluation limits of using solely Dirichlet-simulated datasets.
Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
About OpenReview
Hosting a Venue
All Venues
Contact
Sponsors
Donate
FAQ
Terms of Use / Privacy Policy
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview