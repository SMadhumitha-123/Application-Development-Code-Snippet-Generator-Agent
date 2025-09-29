# agent_demo.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class CodeSnippetAgent:
    def __init__(self, model_name="Salesforce/codegen-350M-multi"):
        """
        Initialize the code generation agent.
        Fully CPU compatible without accelerate.
        """
        self.device = torch.device("cpu")  # force CPU

        # Load tokenizer and model from local cache or download if needed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate_code(self, prompt, max_length=150):
        """
        Generate Python code from a natural language prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return code

    def explain_code(self, code_snippet, max_length=200):
        """
        Generate an explanation for a given Python code snippet.
        """
        explanation_prompt = f"Explain this Python code:\n{code_snippet}"
        inputs = self.tokenizer(explanation_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=max_length)
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return explanation


# -----------------------------
# Quick inline demo of usage
# -----------------------------
if __name__ == "__main__":
    # Initialize the agent
    agent = CodeSnippetAgent()

    # Example 1: Generate a code snippet from a prompt
    prompt = "Write a Python function to print 'Hello World'"
    generated_code = agent.generate_code(prompt, max_length=50)
    print("Generated Code:\n", generated_code)

    # Example 2: Explain a code snippet
    code_to_explain = "def hello_world():\n    print('Hello World')"
    explanation = agent.explain_code(code_to_explain, max_length=50)
    print("\nExplanation:\n", explanation)

    # Example 3: Quick inline test (like your previous snippet)
    text = "def add_numbers(a, b):"
    input_ids = agent.tokenizer(text, return_tensors="pt").input_ids.to(agent.device)
    generated_ids = agent.model.generate(input_ids, max_length=50)
    print("\nInline Generated Code:\n", agent.tokenizer.decode(generated_ids[0], skip_special_tokens=True))
