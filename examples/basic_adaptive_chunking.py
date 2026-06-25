from adaptive_chunking import AdaptiveChunker


TEXT = """
# Adaptive Chunking

Adaptive chunking tries several chunking strategies and picks the best one for
the document instead of assuming one splitter works everywhere.

## References

See Section 2.1 for related implementation details. The selector keeps nearby
context with references when possible.

## Result

The output includes the selected strategy, scored metrics, and ordered chunks.
"""


def main() -> None:
    result = AdaptiveChunker().chunk(TEXT, document_id="readme-demo")

    print(f"Selected strategy: {result.strategy_name}")
    print(f"Adaptive score: {result.score:.3f}")
    print("\nMetrics:")
    for metric in result.metrics:
        print(f"- {metric.name}: {metric.value:.3f} (weight={metric.weight:.2f})")

    print("\nChunks:")
    for chunk in result.chunks:
        print(f"[{chunk.index}] chars={chunk.start_char}:{chunk.end_char} {chunk.text[:80]!r}")


if __name__ == "__main__":
    main()

