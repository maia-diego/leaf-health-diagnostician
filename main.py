import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.train import train_model
    print("Successfully imported train_model.")
except ImportError as e:
    print(f"Error importing train_model: {e}")

try:
    from src.evaluate import evaluate_model
    print("Successfully imported evaluate_model.")
except ImportError as e:
    print(f"Error importing evaluate_model: {e}")

import argparse
from src.train import train_model
from src.evaluate import evaluate_model
from src.explain import explain_model

def main():
    parser = argparse.ArgumentParser(description="Leaf Health Diagnostician")
    parser.add_argument("--train", action="store_true", help="Treinar o modelo")
    parser.add_argument("--evaluate", action="store_true", help="Avaliar o modelo")
    parser.add_argument("--explain", action="store_true", help="Explicar as predições")
    args = parser.parse_args()

    if args.train:
        train_model()
    elif args.evaluate:
        evaluate_model()
    elif args.explain:
        explain_model()
    else:
        print("Use --train, --evaluate ou --explain para executar o projeto.")

if __name__ == "__main__":
    main()
