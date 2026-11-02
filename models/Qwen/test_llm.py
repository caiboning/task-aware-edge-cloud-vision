
import json
import time

from llm import LocalLLM


# ==============================
# Main Test
# ==============================

def main():

    llm = LocalLLM()

    print("\n=== Qwen Semantic Task Test ===")
    print("Enter a visual question.")
    print("Type 'exit' to quit.\n")

    while True:

        try:
            question = input(
                "Question: "
            ).strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting test.")
            break

        if question.lower() in ("exit", "quit"):
            print("Exiting test.")
            break

        if not question:
            continue

        start = time.perf_counter()

        try:
            plan = llm.understand_task(question)

            elapsed = time.perf_counter() - start

            print("\nTask Plan:")
            print(
                json.dumps(
                    plan,
                    indent=4,
                    ensure_ascii=False
                )
            )

            print(
                f"\nInference time: {elapsed:.2f}s"
            )

            print("-" * 40)

        except KeyboardInterrupt:
            print("\nTest interrupted.")
            break

        except Exception as error:
            print(f"\nError: {error}")
            print("-" * 40)


if __name__ == "__main__":
    main()
