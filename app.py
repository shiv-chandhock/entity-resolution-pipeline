import sys
from src.pipeline import run_hybrid_resolver_pipeline

def main():
    print("=" * 60)
    print("INTERACTIVE HYBRID ENTITY RESOLVER TERMINAL")
    print("Type 'exit' or 'quit' to close the program.")
    print("=" * 60)

    while True:
        try:
            user_text = input("\nEnter text to resolve: ").strip()
            
            if user_text.lower() in ["exit", "quit"]:
                print("Closing terminal. Goodbye!")
                break
                
            if not user_text:
                continue
                
            print("-" * 50)
            print("Analyzing text and evaluating contexts...")
            print("-" * 50)
            
            results = run_hybrid_resolver_pipeline(user_text)
            
            if not results:
                print("[No distinct entities could be resolved.]")
            else:
                for entity in results:
                    print(f"-> Found Phrase: '{entity['phrase']}'")
                    print(f"   Wikidata Q-ID: {entity['q_id']}")
                    print(f"   Official Name: {entity['name']}")
                    print(f"   Confidence:    {entity['confidence']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nClosing terminal safely.")
            break
        except Exception as e:
            print(f"\n[Execution Error]: {e}")

if __name__ == "__main__":
    main()