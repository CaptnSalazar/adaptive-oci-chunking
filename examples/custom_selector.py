from adaptive_chunking import AdaptiveChunker
from adaptive_chunking.chunkers import DelimiterChunker, SectionAwareChunker, SemanticChunker
from adaptive_chunking.metrics import IntrinsicMetricEvaluator, MetricConfig, MetricWeights
from adaptive_chunking.selector import AdaptiveSelector


TEXT = """
# Product Policy

Users may request exports from the admin console.

---

# Audit Logging

All export requests are recorded. See Section 4.2 for retention details.

---

# Data Retention

Retention windows vary by workspace plan and compliance setting.
"""


def main() -> None:
    selector = AdaptiveSelector(
        chunkers=[
            SectionAwareChunker(min_size=80, max_size=500),
            DelimiterChunker(delimiter="\n---\n", max_size=500),
            SemanticChunker(min_size=80, max_size=500, similarity_threshold=0.08),
        ],
        evaluator=IntrinsicMetricEvaluator(
            MetricConfig(
                target_min_tokens=8,
                target_max_tokens=80,
                weights=MetricWeights(
                    block_integrity=1.4,
                    coverage=1.5,
                    redundancy=0.8,
                ),
            )
        ),
    )
    result = AdaptiveChunker(selector=selector).chunk(TEXT, document_id="policy")

    print(f"Winner: {result.strategy_name} ({result.score:.3f})")
    print("Candidate ranking:")
    for candidate in result.candidates:
        print(f"- {candidate.strategy_name}: {candidate.score:.3f} chunks={len(candidate.chunks)}")


if __name__ == "__main__":
    main()

