from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_name = "distilgpt2"  # small and CPU friendly
    print("Loading model...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Write a one line fantasy quest hook about a hidden temple in the misty forest:"
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=40,
            temperature=0.9,
            top_p=0.9,
            do_sample=True
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    print("\nGenerated:\n")
    print(text)

if __name__ == "__main__":
    main()

