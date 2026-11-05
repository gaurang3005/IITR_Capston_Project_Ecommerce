"""Task 4 transcript: 5 in-scope queries + 1 out-of-scope query that must hit the fallback."""
import config
from rag_core import generate

for q in ["How long is the return window for footwear?", "When will I get my COD refund?",
          "What is the delivery time for metro cities?", "Can I cancel an order after it has shipped?",
          "How much are 500 loyalty points worth?", "Book me a flight to Delhi tomorrow."]:
    r = generate(q, config.COLLECTION_SENTENCE)
    answer = r["answer"].replace("\n", "\n   ")              # keep every answer line indented
    print(f"\nQ: {q}\n   top-1 similarity {r['top_similarity']} vs threshold {r['threshold']}"
          f" -> grounded={r['grounded']}\n   {answer}")
