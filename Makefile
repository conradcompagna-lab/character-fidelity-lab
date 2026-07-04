.PHONY: chunks index ask eval compare test clean

chunks:
	python -m character_lab.chunking --lore-dir data/lore --out data/index/chunks.jsonl

index: chunks
	python -m character_lab.build_index --chunks data/index/chunks.jsonl --out data/index/vector_store_gemini.json --embedder gemini

ask: index
	python -m character_lab.ask --character data/characters/maelor_v2.json --index data/index/vector_store_gemini.json --provider gemini --model gemini-2.5-flash-lite --question "Did Maelor betray the Sunken Temple?"

eval: index
	mkdir -p reports
	python -m character_lab.run_eval --character data/characters/maelor_v2.json --index data/index/vector_store_gemini.json --provider gemini --model gemini-2.5-flash-lite --cases data/evals/eval_cases.jsonl --max-per-category 3 --out reports/v2_gemini.jsonl

compare: index
	mkdir -p reports
	python -m character_lab.run_eval --character data/characters/maelor_v1.json --index data/index/vector_store_gemini.json --provider gemini --model gemini-2.5-flash-lite --cases data/evals/eval_cases.jsonl --max-per-category 3 --out reports/v1_gemini.jsonl
	python -m character_lab.run_eval --character data/characters/maelor_v2.json --index data/index/vector_store_gemini.json --provider gemini --model gemini-2.5-flash-lite --cases data/evals/eval_cases.jsonl --max-per-category 3 --out reports/v2_gemini.jsonl
	python -m character_lab.compare_runs --before reports/v1_gemini.jsonl --after reports/v2_gemini.jsonl --out reports/v1_vs_v2_gemini.md

test:
	python -m unittest discover -s tests

clean:
	rm -f data/index/chunks.jsonl data/index/vector_store_gemini.json
	rm -rf reports
